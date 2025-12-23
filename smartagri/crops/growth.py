"""
Crop Growth Modeling Module

Provides mechanistic and empirical models for simulating crop growth,
including phenology, biomass accumulation, and canopy development.

Models Implemented:
    - WOFOST-based growth model
    - APSIM-style phenology
    - Leaf area index (LAI) dynamics
    - Root growth and water uptake
    - Stress response modeling

Applications:
    - Yield forecasting
    - Irrigation planning
    - Growth stage prediction
    - Stress detection

Example:
    >>> model = GrowthModel(crop="corn")
    >>> result = model.simulate(
    ...     start_date=date(2024, 5, 1),
    ...     end_date=date(2024, 10, 1),
    ...     weather=weather_data,
    ...     soil=soil_data
    ... )
    >>> print(f"Final biomass: {result.biomass[-1]:.0f} kg/ha")
"""

import numpy as np
from typing import (
    Optional,
    Union,
    List,
    Tuple,
    Dict,
    Any,
)
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum, auto
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class GrowthStage(Enum):
    """
    Phenological growth stages (BBCH scale inspired).

    Standard stages for crop development tracking.
    """
    GERMINATION = 0
    EMERGENCE = 10
    LEAF_DEVELOPMENT = 20
    TILLERING = 30
    STEM_ELONGATION = 40
    BOOTING = 50
    HEADING = 60
    FLOWERING = 70
    GRAIN_FILLING = 80
    RIPENING = 90
    SENESCENCE = 95
    HARVEST = 99


@dataclass
class GrowthState:
    """
    Current state of crop growth.

    Attributes:
        date: Current simulation date
        stage: Phenological growth stage
        gdd: Accumulated growing degree days
        biomass: Total aboveground biomass (kg/ha)
        lai: Leaf area index (m2/m2)
        root_depth: Root depth (m)
        grain_yield: Grain yield (kg/ha)
        water_stress: Water stress factor (0-1)
        nitrogen_stress: Nitrogen stress factor (0-1)
    """
    date: date
    stage: GrowthStage
    gdd: float
    biomass: float
    lai: float
    root_depth: float
    grain_yield: float = 0.0
    water_stress: float = 1.0
    nitrogen_stress: float = 1.0


@dataclass
class GrowthParameters:
    """
    Crop-specific growth parameters.

    Attributes:
        t_base: Base temperature for GDD (°C)
        t_opt: Optimal temperature (°C)
        t_max: Maximum temperature (°C)
        gdd_emergence: GDD for emergence
        gdd_flowering: GDD for flowering
        gdd_maturity: GDD for maturity
        rue: Radiation use efficiency (g/MJ)
        sla: Specific leaf area (m2/g)
        harvest_index: Harvest index (grain/biomass)
        root_growth_rate: Root growth rate (m/day)
        max_root_depth: Maximum root depth (m)
        extinction_coef: Light extinction coefficient
    """
    t_base: float = 10.0
    t_opt: float = 25.0
    t_max: float = 35.0
    gdd_emergence: float = 80.0
    gdd_flowering: float = 800.0
    gdd_maturity: float = 1400.0
    rue: float = 3.0
    sla: float = 0.02
    harvest_index: float = 0.45
    root_growth_rate: float = 0.02
    max_root_depth: float = 1.5
    extinction_coef: float = 0.6


@dataclass
class SimulationResult:
    """
    Results from crop growth simulation.

    Contains time series of all state variables.
    """
    dates: List[date]
    stages: List[GrowthStage]
    gdd: np.ndarray
    biomass: np.ndarray
    lai: np.ndarray
    root_depth: np.ndarray
    grain_yield: np.ndarray
    water_stress: np.ndarray
    nitrogen_stress: np.ndarray
    transpiration: np.ndarray
    evaporation: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "dates": [d.isoformat() for d in self.dates],
            "stages": [s.name for s in self.stages],
            "gdd": self.gdd.tolist(),
            "biomass": self.biomass.tolist(),
            "lai": self.lai.tolist(),
            "root_depth": self.root_depth.tolist(),
            "grain_yield": self.grain_yield.tolist(),
            "water_stress": self.water_stress.tolist(),
            "nitrogen_stress": self.nitrogen_stress.tolist(),
            "transpiration": self.transpiration.tolist(),
            "evaporation": self.evaporation.tolist(),
        }


class GrowingDegreeDays:
    """
    Growing Degree Days (GDD) calculation.

    Calculates thermal time accumulation for crop development,
    supporting various calculation methods.

    Example:
        >>> gdd_calc = GrowingDegreeDays(base_temp=10.0)
        >>> daily_gdd = gdd_calc.calculate(t_max, t_min)
        >>> cumulative_gdd = gdd_calc.cumulative(t_max, t_min)
    """

    def __init__(
        self,
        base_temp: float = 10.0,
        upper_temp: float = 30.0,
        method: str = "average",
    ):
        """
        Initialize GDD calculator.

        Args:
            base_temp: Base temperature for growth (°C)
            upper_temp: Upper temperature threshold (°C)
            method: Calculation method ('average', 'modified', 'sine')
        """
        self.base_temp = base_temp
        self.upper_temp = upper_temp
        self.method = method

    def calculate(
        self,
        t_max: np.ndarray,
        t_min: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate daily GDD values.

        Args:
            t_max: Daily maximum temperatures (°C)
            t_min: Daily minimum temperatures (°C)

        Returns:
            Array of daily GDD values
        """
        t_max = np.asarray(t_max)
        t_min = np.asarray(t_min)

        if self.method == "average":
            # Simple average method
            t_avg = (t_max + t_min) / 2
            gdd = np.maximum(t_avg - self.base_temp, 0)
        elif self.method == "modified":
            # Modified method with upper threshold
            t_max_adj = np.minimum(t_max, self.upper_temp)
            t_min_adj = np.maximum(t_min, self.base_temp)
            t_min_adj = np.minimum(t_min_adj, t_max_adj)
            t_avg = (t_max_adj + t_min_adj) / 2
            gdd = np.maximum(t_avg - self.base_temp, 0)
        else:
            # Default to average method
            t_avg = (t_max + t_min) / 2
            gdd = np.maximum(t_avg - self.base_temp, 0)

        return gdd

    def cumulative(
        self,
        t_max: np.ndarray,
        t_min: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate cumulative GDD.

        Args:
            t_max: Daily maximum temperatures (°C)
            t_min: Daily minimum temperatures (°C)

        Returns:
            Array of cumulative GDD values
        """
        daily_gdd = self.calculate(t_max, t_min)
        return np.cumsum(daily_gdd)

    def days_to_accumulate(
        self,
        target_gdd: float,
        t_max: np.ndarray,
        t_min: np.ndarray,
        start_idx: int = 0,
    ) -> Optional[int]:
        """
        Calculate days to reach target GDD.

        Args:
            target_gdd: Target GDD accumulation
            t_max: Daily maximum temperatures
            t_min: Daily minimum temperatures
            start_idx: Starting index

        Returns:
            Number of days or None if not reached
        """
        cumulative = self.cumulative(t_max[start_idx:], t_min[start_idx:])
        indices = np.where(cumulative >= target_gdd)[0]

        if len(indices) > 0:
            return int(indices[0]) + 1
        return None


class BiomassModel:
    """
    Biomass accumulation model based on radiation use efficiency.

    Calculates daily biomass production from solar radiation and LAI.

    Example:
        >>> model = BiomassModel(radiation_use_efficiency=3.5)
        >>> daily = model.daily_accumulation(solar_radiation, lai)
    """

    def __init__(
        self,
        radiation_use_efficiency: float = 3.0,
        extinction_coefficient: float = 0.6,
    ):
        """
        Initialize biomass model.

        Args:
            radiation_use_efficiency: RUE (g/MJ)
            extinction_coefficient: Light extinction coefficient
        """
        self.rue = radiation_use_efficiency
        self.extinction_coef = extinction_coefficient

    def intercepted_radiation(
        self,
        solar_radiation: np.ndarray,
        lai: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate intercepted photosynthetically active radiation.

        Args:
            solar_radiation: Daily solar radiation (MJ/m²/day)
            lai: Leaf area index

        Returns:
            Intercepted PAR (MJ/m²/day)
        """
        solar_radiation = np.asarray(solar_radiation)
        lai = np.asarray(lai)

        # PAR is ~50% of total solar radiation
        par = solar_radiation * 0.5

        # Beer-Lambert law
        f_int = 1 - np.exp(-self.extinction_coef * lai)

        return par * f_int

    def daily_accumulation(
        self,
        solar_radiation: np.ndarray,
        lai: np.ndarray,
        stress_factor: Union[float, np.ndarray] = 1.0,
    ) -> np.ndarray:
        """
        Calculate daily biomass accumulation.

        Args:
            solar_radiation: Daily solar radiation (MJ/m²/day)
            lai: Leaf area index
            stress_factor: Combined stress factor (0-1)

        Returns:
            Daily biomass accumulation (kg/ha)
        """
        ipar = self.intercepted_radiation(solar_radiation, lai)

        # Stress-adjusted RUE
        effective_rue = self.rue * stress_factor

        # Convert g/m² to kg/ha (multiply by 10)
        biomass = ipar * effective_rue * 10

        return np.maximum(biomass, 0)

    def total_biomass(
        self,
        solar_radiation: np.ndarray,
        lai: np.ndarray,
        stress_factor: Union[float, np.ndarray] = 1.0,
    ) -> float:
        """
        Calculate total accumulated biomass.

        Args:
            solar_radiation: Daily solar radiation (MJ/m²/day)
            lai: Leaf area index
            stress_factor: Combined stress factor (0-1)

        Returns:
            Total biomass (kg/ha)
        """
        daily = self.daily_accumulation(solar_radiation, lai, stress_factor)
        return float(np.sum(daily))


class PhenologyModel:
    """
    Crop phenology model based on thermal time.

    Predicts growth stage transitions using growing degree days
    with optional photoperiod sensitivity.

    Example:
        >>> phenology = PhenologyModel(crop_type="wheat")
        >>> stage = phenology.get_stage(gdd=500)
        >>> days_to_flower = phenology.days_to_stage(
        ...     current_gdd=300,
        ...     target_stage=GrowthStage.FLOWERING,
        ...     avg_daily_gdd=15
        ... )
    """

    def __init__(
        self,
        crop: str = "generic",
        crop_type: str = None,
        params: Optional[GrowthParameters] = None,
    ):
        """
        Initialize phenology model.

        Args:
            crop: Crop name
            crop_type: Alternative crop name parameter
            params: Growth parameters (uses defaults if None)
        """
        self.crop = crop_type if crop_type is not None else crop
        self.params = params or GrowthParameters()

        # GDD thresholds for stage transitions
        self._stage_thresholds = {
            GrowthStage.GERMINATION: 0,
            GrowthStage.EMERGENCE: self.params.gdd_emergence,
            GrowthStage.LEAF_DEVELOPMENT: self.params.gdd_emergence * 1.5,
            GrowthStage.TILLERING: self.params.gdd_flowering * 0.3,
            GrowthStage.STEM_ELONGATION: self.params.gdd_flowering * 0.5,
            GrowthStage.BOOTING: self.params.gdd_flowering * 0.7,
            GrowthStage.HEADING: self.params.gdd_flowering * 0.85,
            GrowthStage.FLOWERING: self.params.gdd_flowering,
            GrowthStage.GRAIN_FILLING: self.params.gdd_flowering * 1.3,
            GrowthStage.RIPENING: self.params.gdd_maturity * 0.9,
            GrowthStage.SENESCENCE: self.params.gdd_maturity,
            GrowthStage.HARVEST: self.params.gdd_maturity * 1.1,
        }

    def get_stage(self, gdd: float, return_enum: bool = False) -> Union[str, GrowthStage]:
        """
        Get growth stage for given GDD.

        Args:
            gdd: Accumulated growing degree days
            return_enum: If True, return GrowthStage enum; else return string

        Returns:
            Current growth stage (string or enum)
        """
        current_stage = GrowthStage.GERMINATION

        for stage, threshold in self._stage_thresholds.items():
            if gdd >= threshold:
                current_stage = stage

        if return_enum:
            return current_stage
        return current_stage.name.lower().replace('_', ' ')

    def stage_progress(self, gdd: float) -> Tuple[GrowthStage, float]:
        """
        Get stage and progress within stage.

        Args:
            gdd: Accumulated GDD

        Returns:
            Tuple of (stage, progress_fraction)
        """
        current_stage = self.get_stage(gdd)
        stage_values = list(self._stage_thresholds.values())
        stage_keys = list(self._stage_thresholds.keys())

        current_idx = stage_keys.index(current_stage)
        current_threshold = stage_values[current_idx]

        if current_idx < len(stage_values) - 1:
            next_threshold = stage_values[current_idx + 1]
            if next_threshold > current_threshold:
                progress = (gdd - current_threshold) / (next_threshold - current_threshold)
                return current_stage, min(progress, 1.0)

        return current_stage, 1.0

    def days_to_stage(
        self,
        current_gdd: float,
        target_stage: GrowthStage,
        avg_daily_gdd: float,
    ) -> Optional[int]:
        """
        Estimate days until target stage.

        Args:
            current_gdd: Current accumulated GDD
            target_stage: Target growth stage
            avg_daily_gdd: Average daily GDD accumulation

        Returns:
            Estimated days or None if already past stage
        """
        target_gdd = self._stage_thresholds.get(target_stage)

        if target_gdd is None or current_gdd >= target_gdd:
            return None

        gdd_remaining = target_gdd - current_gdd
        return int(np.ceil(gdd_remaining / avg_daily_gdd))

    def calculate_gdd(
        self,
        t_max: float,
        t_min: float,
        method: str = "average",
    ) -> float:
        """
        Calculate daily GDD.

        Args:
            t_max: Maximum temperature (°C)
            t_min: Minimum temperature (°C)
            method: Calculation method ('average', 'modified')

        Returns:
            Daily GDD
        """
        t_base = self.params.t_base

        if method == "average":
            t_avg = (t_max + t_min) / 2
            return max(t_avg - t_base, 0)
        else:  # modified
            t_max = min(t_max, self.params.t_max)
            t_min = max(t_min, t_base)
            if t_min > t_max:
                t_min = t_max
            t_avg = (t_max + t_min) / 2
            return max(t_avg - t_base, 0)


class BiomassAccumulation:
    """
    Biomass accumulation model based on radiation use efficiency.

    Calculates daily biomass production from intercepted
    photosynthetically active radiation (PAR).

    Example:
        >>> model = BiomassAccumulation(rue=3.5)
        >>> daily_growth = model.calculate(
        ...     radiation=15.0,
        ...     lai=3.0,
        ...     water_stress=0.8
        ... )
    """

    def __init__(
        self,
        rue: float = 3.0,
        extinction_coef: float = 0.6,
    ):
        """
        Initialize biomass accumulation model.

        Args:
            rue: Radiation use efficiency (g/MJ)
            extinction_coef: Light extinction coefficient
        """
        self.rue = rue
        self.extinction_coef = extinction_coef

    def intercepted_par(
        self,
        radiation: float,
        lai: float,
    ) -> float:
        """
        Calculate intercepted PAR.

        Uses Beer-Lambert law for canopy light interception.

        Args:
            radiation: Daily solar radiation (MJ/m2/day)
            lai: Leaf area index (m2/m2)

        Returns:
            Intercepted PAR (MJ/m2/day)
        """
        # PAR is approximately 50% of solar radiation
        par = radiation * 0.5

        # Beer-Lambert interception
        f_int = 1 - np.exp(-self.extinction_coef * lai)

        return par * f_int

    def calculate(
        self,
        radiation: float,
        lai: float,
        water_stress: float = 1.0,
        nitrogen_stress: float = 1.0,
        temperature_factor: float = 1.0,
    ) -> float:
        """
        Calculate daily biomass production.

        Args:
            radiation: Daily solar radiation (MJ/m2/day)
            lai: Leaf area index
            water_stress: Water stress factor (0-1)
            nitrogen_stress: Nitrogen stress factor (0-1)
            temperature_factor: Temperature stress factor (0-1)

        Returns:
            Daily biomass production (kg/ha)
        """
        # Intercepted PAR
        ipar = self.intercepted_par(radiation, lai)

        # Stress-adjusted RUE
        stress_factor = min(water_stress, nitrogen_stress, temperature_factor)
        effective_rue = self.rue * stress_factor

        # Biomass production (g/m2 to kg/ha)
        biomass = ipar * effective_rue * 10  # Convert g/m2 to kg/ha

        return max(biomass, 0)

    def temperature_response(
        self,
        temperature: float,
        t_base: float = 10.0,
        t_opt: float = 25.0,
        t_max: float = 35.0,
    ) -> float:
        """
        Calculate temperature response factor.

        Args:
            temperature: Air temperature (°C)
            t_base: Base temperature
            t_opt: Optimal temperature
            t_max: Maximum temperature

        Returns:
            Temperature factor (0-1)
        """
        if temperature <= t_base or temperature >= t_max:
            return 0.0
        elif temperature <= t_opt:
            return (temperature - t_base) / (t_opt - t_base)
        else:
            return (t_max - temperature) / (t_max - t_opt)


class LeafAreaIndex:
    """
    Leaf Area Index (LAI) dynamics model.

    Models LAI development from emergence to senescence,
    including leaf expansion, maintenance, and death.

    Example:
        >>> lai_model = LeafAreaIndex(max_lai=6.0)
        >>> lai = lai_model.calculate(gdd=500, biomass=5000)
    """

    def __init__(
        self,
        max_lai: float = 6.0,
        sla: float = 0.02,
        leaf_fraction_max: float = 0.6,
    ):
        """
        Initialize LAI model.

        Args:
            max_lai: Maximum LAI (m2/m2)
            sla: Specific leaf area (m2/g)
            leaf_fraction_max: Maximum fraction of biomass as leaves
        """
        self.max_lai = max_lai
        self.sla = sla
        self.leaf_fraction_max = leaf_fraction_max

    def calculate(
        self,
        gdd: float,
        biomass: float,
        gdd_flowering: float = 800.0,
        gdd_maturity: float = 1400.0,
    ) -> float:
        """
        Calculate LAI from GDD and biomass.

        Args:
            gdd: Accumulated GDD
            biomass: Aboveground biomass (kg/ha)
            gdd_flowering: GDD at flowering
            gdd_maturity: GDD at maturity

        Returns:
            Leaf area index (m2/m2)
        """
        # Leaf fraction varies with phenology
        if gdd < gdd_flowering:
            # Increasing leaf fraction during vegetative phase
            leaf_fraction = self.leaf_fraction_max * min(gdd / gdd_flowering, 1.0)
        else:
            # Declining leaf fraction during reproductive phase
            senescence_progress = (gdd - gdd_flowering) / (gdd_maturity - gdd_flowering)
            leaf_fraction = self.leaf_fraction_max * (1 - 0.8 * min(senescence_progress, 1.0))

        # Calculate LAI from leaf biomass
        leaf_biomass = biomass * leaf_fraction  # kg/ha
        leaf_biomass_g_m2 = leaf_biomass / 10  # Convert to g/m2

        lai = leaf_biomass_g_m2 * self.sla

        return min(lai, self.max_lai)

    def senescence_factor(
        self,
        gdd: float,
        gdd_maturity: float,
    ) -> float:
        """
        Calculate leaf senescence factor.

        Args:
            gdd: Current GDD
            gdd_maturity: GDD at maturity

        Returns:
            Senescence factor (0-1, 1 = no senescence)
        """
        if gdd < gdd_maturity * 0.7:
            return 1.0
        elif gdd > gdd_maturity:
            return 0.1
        else:
            progress = (gdd - gdd_maturity * 0.7) / (gdd_maturity * 0.3)
            return 1.0 - 0.9 * progress


class RootGrowthModel:
    """
    Root growth and water uptake model.

    Simulates root zone expansion and water extraction
    from soil layers.

    Example:
        >>> root_model = RootGrowthModel(max_depth=1.5)
        >>> depth = root_model.depth_at_gdd(gdd=600)
        >>> uptake = root_model.water_uptake(soil_moisture, root_density)
    """

    def __init__(
        self,
        max_depth: float = 1.5,
        growth_rate: float = 0.02,
        initial_depth: float = 0.05,
    ):
        """
        Initialize root growth model.

        Args:
            max_depth: Maximum rooting depth (m)
            growth_rate: Root growth rate (m/day)
            initial_depth: Initial root depth (m)
        """
        self.max_depth = max_depth
        self.growth_rate = growth_rate
        self.initial_depth = initial_depth

    def depth_at_gdd(
        self,
        gdd: float,
        gdd_per_day: float = 15.0,
    ) -> float:
        """
        Calculate root depth at given GDD.

        Args:
            gdd: Accumulated GDD
            gdd_per_day: Average daily GDD

        Returns:
            Root depth (m)
        """
        # Convert GDD to approximate days
        days = gdd / gdd_per_day

        # Root growth with diminishing rate
        depth = self.initial_depth + self.growth_rate * days * (
            1 - self.initial_depth / self.max_depth
        )

        return min(depth, self.max_depth)

    def root_density_profile(
        self,
        depth: float,
        n_layers: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate root density distribution with depth.

        Args:
            depth: Current root depth (m)
            n_layers: Number of soil layers

        Returns:
            Tuple of (layer_depths, root_densities)
        """
        layer_depths = np.linspace(0, depth, n_layers)
        layer_thickness = depth / n_layers

        # Exponential root distribution
        decay_rate = 3.0 / depth  # Most roots in upper 1/3
        root_density = np.exp(-decay_rate * layer_depths)

        # Normalize
        root_density = root_density / np.sum(root_density)

        return layer_depths, root_density

    def water_uptake(
        self,
        soil_moisture: np.ndarray,
        root_density: np.ndarray,
        potential_transpiration: float,
        wilting_point: float = 0.15,
        field_capacity: float = 0.35,
    ) -> float:
        """
        Calculate water uptake from soil layers.

        Args:
            soil_moisture: Volumetric water content per layer
            root_density: Root density per layer
            potential_transpiration: Potential transpiration (mm/day)
            wilting_point: Wilting point water content
            field_capacity: Field capacity water content

        Returns:
            Actual water uptake (mm/day)
        """
        # Available water fraction per layer
        available = np.clip(
            (soil_moisture - wilting_point) / (field_capacity - wilting_point),
            0, 1
        )

        # Weighted uptake capacity
        uptake_capacity = np.sum(available * root_density)

        # Actual uptake
        actual_uptake = potential_transpiration * uptake_capacity

        return actual_uptake


class GrowthModel:
    """
    Integrated crop growth model.

    Combines phenology, biomass, LAI, and root models for
    complete crop growth simulation.

    Example:
        >>> model = GrowthModel(crop="corn")
        >>> result = model.simulate(
        ...     start_date=date(2024, 5, 1),
        ...     end_date=date(2024, 10, 1),
        ...     weather=weather_df,
        ...     soil=soil_params
        ... )
        >>> final_yield = result.grain_yield[-1]
    """

    def __init__(
        self,
        crop: str = "corn",
        params: Optional[GrowthParameters] = None,
        use_gpu: bool = False,
    ):
        """
        Initialize growth model.

        Args:
            crop: Crop name
            params: Growth parameters
            use_gpu: Use GPU acceleration
        """
        self.crop = crop
        self.params = params or self._default_params(crop)
        self.use_gpu = use_gpu

        # Initialize sub-models
        self.phenology = PhenologyModel(crop, self.params)
        self.biomass_model = BiomassAccumulation(
            rue=self.params.rue,
            extinction_coef=self.params.extinction_coef,
        )
        self.lai_model = LeafAreaIndex(sla=self.params.sla)
        self.root_model = RootGrowthModel(
            max_depth=self.params.max_root_depth,
            growth_rate=self.params.root_growth_rate,
        )

    def _default_params(self, crop: str) -> GrowthParameters:
        """Get default parameters for crop."""
        defaults = {
            "corn": GrowthParameters(
                t_base=10, t_opt=28, t_max=38,
                gdd_emergence=80, gdd_flowering=850, gdd_maturity=1500,
                rue=3.8, harvest_index=0.48,
            ),
            "wheat": GrowthParameters(
                t_base=0, t_opt=20, t_max=30,
                gdd_emergence=100, gdd_flowering=900, gdd_maturity=1800,
                rue=2.8, harvest_index=0.42,
            ),
            "soybean": GrowthParameters(
                t_base=10, t_opt=28, t_max=35,
                gdd_emergence=90, gdd_flowering=700, gdd_maturity=1400,
                rue=2.5, harvest_index=0.35,
            ),
            "rice": GrowthParameters(
                t_base=12, t_opt=28, t_max=38,
                gdd_emergence=80, gdd_flowering=800, gdd_maturity=1600,
                rue=2.6, harvest_index=0.45,
            ),
        }
        return defaults.get(crop, GrowthParameters())

    def simulate(
        self,
        start_date: date,
        end_date: date,
        weather: Dict[str, np.ndarray],
        soil: Optional[Dict[str, Any]] = None,
    ) -> SimulationResult:
        """
        Run crop growth simulation.

        Args:
            start_date: Planting date
            end_date: End of simulation
            weather: Weather data with keys 't_max', 't_min', 'radiation', 'precipitation'
            soil: Soil parameters (optional)

        Returns:
            SimulationResult with time series of state variables
        """
        n_days = (end_date - start_date).days + 1

        # Initialize output arrays
        dates = [start_date + timedelta(days=i) for i in range(n_days)]
        stages = []
        gdd = np.zeros(n_days)
        biomass = np.zeros(n_days)
        lai = np.zeros(n_days)
        root_depth = np.zeros(n_days)
        grain_yield = np.zeros(n_days)
        water_stress = np.ones(n_days)
        nitrogen_stress = np.ones(n_days)
        transpiration = np.zeros(n_days)
        evaporation = np.zeros(n_days)

        # Initial state
        current_gdd = 0.0
        current_biomass = 0.0
        current_lai = 0.0
        current_root_depth = self.root_model.initial_depth

        # Soil water (simplified)
        if soil is not None:
            soil_water = soil.get("initial_water", 200.0)  # mm
            field_capacity = soil.get("field_capacity", 300.0)  # mm
        else:
            soil_water = 200.0
            field_capacity = 300.0

        # Daily simulation
        for i in range(n_days):
            # Get weather for day
            t_max = weather["t_max"][i] if i < len(weather["t_max"]) else 25.0
            t_min = weather["t_min"][i] if i < len(weather["t_min"]) else 15.0
            radiation = weather.get("radiation", np.full(n_days, 20.0))[i]
            precip = weather.get("precipitation", np.zeros(n_days))[i]

            # Calculate daily GDD
            daily_gdd = self.phenology.calculate_gdd(t_max, t_min)
            current_gdd += daily_gdd

            # Get growth stage
            stage = self.phenology.get_stage(current_gdd)

            # Calculate water stress
            water_frac = soil_water / field_capacity
            ws = min(1.0, max(0.2, water_frac / 0.5))

            # Temperature factor
            t_avg = (t_max + t_min) / 2
            tf = self.biomass_model.temperature_response(
                t_avg, self.params.t_base, self.params.t_opt, self.params.t_max
            )

            # Calculate LAI
            current_lai = self.lai_model.calculate(
                current_gdd, current_biomass,
                self.params.gdd_flowering, self.params.gdd_maturity
            )

            # Calculate biomass growth
            if stage.value >= GrowthStage.EMERGENCE.value:
                daily_biomass = self.biomass_model.calculate(
                    radiation, current_lai, ws, 1.0, tf
                )
                current_biomass += daily_biomass

            # Root growth
            current_root_depth = self.root_model.depth_at_gdd(current_gdd)

            # Grain filling
            if stage.value >= GrowthStage.GRAIN_FILLING.value:
                # Proportion of biomass going to grain
                fill_progress = min(1.0, (current_gdd - self.params.gdd_flowering) /
                                    (self.params.gdd_maturity - self.params.gdd_flowering))
                current_grain = current_biomass * self.params.harvest_index * fill_progress
            else:
                current_grain = 0.0

            # Water balance (simplified)
            pet = 5.0  # Potential ET mm/day (simplified)
            transpiration_today = pet * min(current_lai / 3.0, 1.0) * ws
            evaporation_today = pet * (1 - min(current_lai / 3.0, 1.0)) * 0.5
            soil_water = soil_water + precip - transpiration_today - evaporation_today
            soil_water = max(0, min(soil_water, field_capacity * 1.2))

            # Store results
            stages.append(stage)
            gdd[i] = current_gdd
            biomass[i] = current_biomass
            lai[i] = current_lai
            root_depth[i] = current_root_depth
            grain_yield[i] = current_grain
            water_stress[i] = ws
            transpiration[i] = transpiration_today
            evaporation[i] = evaporation_today

        return SimulationResult(
            dates=dates,
            stages=stages,
            gdd=gdd,
            biomass=biomass,
            lai=lai,
            root_depth=root_depth,
            grain_yield=grain_yield,
            water_stress=water_stress,
            nitrogen_stress=nitrogen_stress,
            transpiration=transpiration,
            evaporation=evaporation,
        )

    def predict_yield(
        self,
        weather: Dict[str, np.ndarray],
        soil: Optional[Dict[str, Any]] = None,
        planting_date: Optional[date] = None,
    ) -> float:
        """
        Predict final grain yield.

        Args:
            weather: Season weather data
            soil: Soil parameters
            planting_date: Planting date

        Returns:
            Predicted yield (kg/ha)
        """
        if planting_date is None:
            planting_date = date.today()

        n_days = len(weather.get("t_max", []))
        end_date = planting_date + timedelta(days=n_days - 1)

        result = self.simulate(planting_date, end_date, weather, soil)

        return result.grain_yield[-1]
