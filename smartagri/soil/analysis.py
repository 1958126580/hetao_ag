"""
Soil Analysis Module

Provides comprehensive soil property analysis including:
- Texture classification (USDA system)
- Hydraulic property estimation
- Bulk density and porosity
- Water retention curves

Example:
    >>> analyzer = SoilAnalyzer()
    >>> props = analyzer.estimate_properties(sand=45, silt=30, clay=25)
    >>> print(f"Texture: {props.texture_class}, FC: {props.field_capacity:.2f}")
"""

import numpy as np
from typing import Optional, Dict, Tuple, List, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SoilProperties:
    """
    Comprehensive soil property container.

    Attributes:
        sand: Sand content (%)
        silt: Silt content (%)
        clay: Clay content (%)
        texture_class: USDA texture class
        bulk_density: Bulk density (g/cm3)
        porosity: Total porosity (m3/m3)
        field_capacity: Volumetric water at field capacity
        wilting_point: Volumetric water at wilting point
        available_water: Plant available water (mm/m)
        saturated_conductivity: Saturated hydraulic conductivity (cm/day)
        organic_matter: Organic matter content (%)
        ph: Soil pH
        cec: Cation exchange capacity (meq/100g)
    """
    sand: float
    silt: float
    clay: float
    texture_class: str
    bulk_density: float
    porosity: float
    field_capacity: float
    wilting_point: float
    available_water: float
    saturated_conductivity: float
    organic_matter: float = 2.5
    ph: float = 6.5
    cec: float = 15.0


class TextureClassifier:
    """
    USDA soil texture classification.

    Classifies soil based on sand, silt, and clay percentages
    using the USDA texture triangle.

    Example:
        >>> classifier = TextureClassifier()
        >>> texture = classifier.classify(sand=45, silt=35, clay=20)
        >>> print(texture)  # "loam"
    """

    # Texture class boundaries (simplified polygon definitions)
    _texture_classes = {
        "sand": lambda s, si, c: s >= 85 and c < 10,
        "loamy_sand": lambda s, si, c: 70 <= s < 85 and c < 15,
        "sandy_loam": lambda s, si, c: (50 <= s < 70 and c < 20) or (s >= 43 and c < 7),
        "loam": lambda s, si, c: 23 <= s < 52 and 28 <= si < 50 and 7 <= c < 27,
        "silt_loam": lambda s, si, c: (si >= 50 and c < 27) or (50 <= si < 80 and c < 12),
        "silt": lambda s, si, c: si >= 80 and c < 12,
        "sandy_clay_loam": lambda s, si, c: 45 <= s < 80 and 20 <= c < 35,
        "clay_loam": lambda s, si, c: 20 <= s < 45 and 15 <= si < 53 and 27 <= c < 40,
        "silty_clay_loam": lambda s, si, c: si >= 40 and 27 <= c < 40,
        "sandy_clay": lambda s, si, c: s >= 45 and c >= 35,
        "silty_clay": lambda s, si, c: si >= 40 and c >= 40,
        "clay": lambda s, si, c: c >= 40 and s < 45 and si < 40,
    }

    def classify(
        self,
        sand: float,
        silt: float,
        clay: float,
    ) -> str:
        """
        Classify soil texture.

        Args:
            sand: Sand percentage (0-100)
            silt: Silt percentage (0-100)
            clay: Clay percentage (0-100)

        Returns:
            USDA texture class name

        Raises:
            ValueError: If percentages don't sum to ~100
        """
        total = sand + silt + clay
        if abs(total - 100) > 1:
            raise ValueError(f"Texture fractions must sum to 100, got {total}")

        # Normalize
        sand = sand / total * 100
        silt = silt / total * 100
        clay = clay / total * 100

        for texture_name, check_func in self._texture_classes.items():
            if check_func(sand, silt, clay):
                return texture_name

        # Default fallback using triangle regions
        if clay >= 40:
            return "clay"
        elif silt >= 50:
            return "silt_loam"
        elif sand >= 50:
            return "sandy_loam"
        else:
            return "loam"

    def get_texture_properties(self, texture_class: str) -> Dict[str, float]:
        """
        Get typical properties for a texture class.

        Args:
            texture_class: USDA texture class

        Returns:
            Dict of typical property values
        """
        properties = {
            "sand": {"ksat": 500, "fc": 0.12, "wp": 0.04, "bd": 1.6},
            "loamy_sand": {"ksat": 350, "fc": 0.15, "wp": 0.05, "bd": 1.55},
            "sandy_loam": {"ksat": 100, "fc": 0.22, "wp": 0.08, "bd": 1.5},
            "loam": {"ksat": 25, "fc": 0.30, "wp": 0.12, "bd": 1.4},
            "silt_loam": {"ksat": 15, "fc": 0.35, "wp": 0.13, "bd": 1.35},
            "silt": {"ksat": 10, "fc": 0.38, "wp": 0.15, "bd": 1.3},
            "sandy_clay_loam": {"ksat": 10, "fc": 0.32, "wp": 0.15, "bd": 1.45},
            "clay_loam": {"ksat": 5, "fc": 0.37, "wp": 0.20, "bd": 1.35},
            "silty_clay_loam": {"ksat": 3, "fc": 0.40, "wp": 0.22, "bd": 1.3},
            "sandy_clay": {"ksat": 2, "fc": 0.35, "wp": 0.23, "bd": 1.4},
            "silty_clay": {"ksat": 1.5, "fc": 0.42, "wp": 0.27, "bd": 1.25},
            "clay": {"ksat": 1, "fc": 0.45, "wp": 0.30, "bd": 1.2},
        }

        return properties.get(texture_class, properties["loam"])


class WaterRetention:
    """
    Soil water retention modeling using van Genuchten equations.

    Models the relationship between soil water content and
    matric potential for irrigation and drainage calculations.

    Example:
        >>> retention = WaterRetention(theta_r=0.05, theta_s=0.45, alpha=0.02, n=1.5)
        >>> theta = retention.theta(psi=-100)
    """

    def __init__(
        self,
        theta_r: float = 0.05,
        theta_s: float = 0.45,
        alpha: float = 0.02,
        n: float = 1.5,
    ):
        """
        Initialize water retention model.

        Args:
            theta_r: Residual water content
            theta_s: Saturated water content
            alpha: van Genuchten alpha parameter (1/cm)
            n: van Genuchten n parameter
        """
        self._theta_r = theta_r
        self._theta_s = theta_s
        self._alpha = alpha
        self._n = n

    def theta(self, psi: float) -> float:
        """
        Calculate water content at given matric potential using instance parameters.

        Args:
            psi: Matric potential (cm, negative for unsaturated)

        Returns:
            Volumetric water content (m3/m3)
        """
        return self.water_content(
            psi, self._theta_r, self._theta_s, self._alpha, self._n
        )

    # Van Genuchten parameters by texture (Carsel & Parrish, 1988) - class variable
    _vg_params = {
            "sand": {"theta_r": 0.045, "theta_s": 0.43, "alpha": 0.145, "n": 2.68},
            "loamy_sand": {"theta_r": 0.057, "theta_s": 0.41, "alpha": 0.124, "n": 2.28},
            "sandy_loam": {"theta_r": 0.065, "theta_s": 0.41, "alpha": 0.075, "n": 1.89},
            "loam": {"theta_r": 0.078, "theta_s": 0.43, "alpha": 0.036, "n": 1.56},
            "silt_loam": {"theta_r": 0.067, "theta_s": 0.45, "alpha": 0.020, "n": 1.41},
            "silt": {"theta_r": 0.034, "theta_s": 0.46, "alpha": 0.016, "n": 1.37},
            "sandy_clay_loam": {"theta_r": 0.100, "theta_s": 0.39, "alpha": 0.059, "n": 1.48},
            "clay_loam": {"theta_r": 0.095, "theta_s": 0.41, "alpha": 0.019, "n": 1.31},
            "silty_clay_loam": {"theta_r": 0.089, "theta_s": 0.43, "alpha": 0.010, "n": 1.23},
            "sandy_clay": {"theta_r": 0.100, "theta_s": 0.38, "alpha": 0.027, "n": 1.23},
            "silty_clay": {"theta_r": 0.070, "theta_s": 0.36, "alpha": 0.005, "n": 1.09},
            "clay": {"theta_r": 0.068, "theta_s": 0.38, "alpha": 0.008, "n": 1.09},
        }

    def estimate_parameters(
        self,
        sand: float,
        silt: float,
        clay: float,
        organic_matter: float = 2.5,
    ) -> Dict[str, float]:
        """
        Estimate van Genuchten parameters from soil texture.

        Uses pedotransfer functions to estimate water retention parameters.

        Args:
            sand: Sand content (%)
            silt: Silt content (%)
            clay: Clay content (%)
            organic_matter: Organic matter content (%)

        Returns:
            Dict with van Genuchten parameters
        """
        # Classify texture
        classifier = TextureClassifier()
        texture = classifier.classify(sand, silt, clay)

        # Get base parameters
        params = dict(self._vg_params.get(texture, self._vg_params["loam"]))

        # Adjust for organic matter
        om_effect = (organic_matter - 2.5) * 0.02
        params["theta_s"] = min(0.55, params["theta_s"] + om_effect)
        params["theta_r"] = max(0.01, params["theta_r"] - om_effect * 0.5)

        return params

    def water_content(
        self,
        psi: float,
        theta_r: float,
        theta_s: float,
        alpha: float,
        n: float,
    ) -> float:
        """
        Calculate water content at given matric potential.

        Uses van Genuchten equation:
        θ(ψ) = θr + (θs - θr) / [1 + (α|ψ|)^n]^m

        Args:
            psi: Matric potential (cm, negative for unsaturated)
            theta_r: Residual water content
            theta_s: Saturated water content
            alpha: Inverse of air entry pressure (1/cm)
            n: Pore size distribution parameter

        Returns:
            Volumetric water content (m3/m3)
        """
        if psi >= 0:
            return theta_s

        m = 1 - 1 / n
        h = abs(psi)

        se = 1 / (1 + (alpha * h) ** n) ** m
        theta = theta_r + (theta_s - theta_r) * se

        return theta

    def matric_potential(
        self,
        theta: float,
        theta_r: float,
        theta_s: float,
        alpha: float,
        n: float,
    ) -> float:
        """
        Calculate matric potential at given water content.

        Inverse of van Genuchten equation.

        Args:
            theta: Volumetric water content
            theta_r, theta_s, alpha, n: Van Genuchten parameters

        Returns:
            Matric potential (cm, negative)
        """
        if theta >= theta_s:
            return 0

        m = 1 - 1 / n
        se = (theta - theta_r) / (theta_s - theta_r)
        se = max(0.001, min(0.999, se))

        h = (1 / alpha) * ((1 / se) ** (1 / m) - 1) ** (1 / n)

        return -h

    def hydraulic_conductivity(
        self,
        theta: float,
        theta_r: float,
        theta_s: float,
        alpha: float,
        n: float,
        k_sat: float,
    ) -> float:
        """
        Calculate unsaturated hydraulic conductivity.

        Uses van Genuchten-Mualem equation.

        Args:
            theta: Volumetric water content
            theta_r, theta_s, alpha, n: Van Genuchten parameters
            k_sat: Saturated hydraulic conductivity (cm/day)

        Returns:
            Unsaturated hydraulic conductivity (cm/day)
        """
        if theta >= theta_s:
            return k_sat

        m = 1 - 1 / n
        se = (theta - theta_r) / (theta_s - theta_r)
        se = max(0.001, min(0.999, se))

        k = k_sat * se ** 0.5 * (1 - (1 - se ** (1 / m)) ** m) ** 2

        return k

    def water_retention_curve(
        self,
        params: Dict[str, float],
        n_points: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate water retention curve.

        Args:
            params: Van Genuchten parameters
            n_points: Number of points

        Returns:
            Tuple of (matric_potential, water_content) arrays
        """
        psi = np.logspace(-1, 5, n_points) * -1  # -0.1 to -100000 cm
        theta = np.array([
            self.water_content(p, **params) for p in psi
        ])

        return psi, theta


class HydraulicConductivity:
    """
    Hydraulic conductivity model.

    Calculates saturated and unsaturated hydraulic conductivity
    based on soil properties and saturation levels.

    Example:
        >>> hc = HydraulicConductivity(k_sat=100.0)
        >>> k = hc.calculate(saturation=0.5)
    """

    def __init__(
        self,
        k_sat: float = 100.0,
        lambda_param: float = 0.5,
    ):
        """
        Initialize hydraulic conductivity model.

        Args:
            k_sat: Saturated hydraulic conductivity (cm/day)
            lambda_param: Pore size distribution parameter
        """
        self.k_sat = k_sat
        self.lambda_param = lambda_param

    def calculate(
        self,
        saturation: float,
        method: str = "brooks_corey",
    ) -> float:
        """
        Calculate hydraulic conductivity at given saturation.

        Args:
            saturation: Effective saturation (0-1)
            method: Calculation method ('brooks_corey', 'van_genuchten')

        Returns:
            Hydraulic conductivity (cm/day)
        """
        saturation = max(0.001, min(1.0, saturation))

        if method == "brooks_corey":
            # Brooks-Corey model
            k = self.k_sat * saturation ** (3 + 2 / self.lambda_param)
        elif method == "van_genuchten":
            # Van Genuchten-Mualem model
            m = self.lambda_param / (1 + self.lambda_param)
            k = self.k_sat * saturation ** 0.5 * (1 - (1 - saturation ** (1/m)) ** m) ** 2
        else:
            # Simple power law
            k = self.k_sat * saturation ** 3

        return max(k, 0.0)

    def relative_conductivity(self, saturation: float) -> float:
        """
        Calculate relative hydraulic conductivity (K/Ksat).

        Args:
            saturation: Effective saturation (0-1)

        Returns:
            Relative conductivity (0-1)
        """
        return self.calculate(saturation) / self.k_sat if self.k_sat > 0 else 0.0


class SoilAnalyzer:
    """
    Comprehensive soil property analyzer.

    Integrates texture classification, water retention modeling,
    and pedotransfer functions for complete soil characterization.

    Example:
        >>> analyzer = SoilAnalyzer()
        >>> properties = analyzer.estimate_properties(
        ...     sand=40, silt=35, clay=25,
        ...     organic_matter=3.0
        ... )
        >>> print(f"Available water: {properties.available_water:.0f} mm/m")
    """

    def __init__(self):
        """Initialize soil analyzer."""
        self.texture_classifier = TextureClassifier()
        self.water_retention = WaterRetention()

    def estimate_properties(
        self,
        sand: float,
        silt: float,
        clay: float,
        organic_matter: float = 2.5,
        bulk_density: Optional[float] = None,
    ) -> SoilProperties:
        """
        Estimate comprehensive soil properties.

        Args:
            sand: Sand content (%)
            silt: Silt content (%)
            clay: Clay content (%)
            organic_matter: Organic matter content (%)
            bulk_density: Bulk density (g/cm3) if known

        Returns:
            SoilProperties with all estimated properties
        """
        # Classify texture
        texture = self.texture_classifier.classify(sand, silt, clay)

        # Get typical properties for texture
        typical = self.texture_classifier.get_texture_properties(texture)

        # Estimate bulk density if not provided
        if bulk_density is None:
            # Adams (1973) pedotransfer function
            bulk_density = typical["bd"]
            # Adjust for organic matter
            bulk_density -= (organic_matter - 2.5) * 0.05

        # Calculate porosity
        particle_density = 2.65  # g/cm3
        porosity = 1 - bulk_density / particle_density

        # Get water retention parameters
        vg_params = self.water_retention.estimate_parameters(
            sand, silt, clay, organic_matter
        )

        # Calculate field capacity (-33 kPa = -340 cm)
        field_capacity = self.water_retention.water_content(
            psi=-340, **vg_params
        )

        # Calculate wilting point (-1500 kPa = -15300 cm)
        wilting_point = self.water_retention.water_content(
            psi=-15300, **vg_params
        )

        # Available water capacity (mm per meter of soil)
        available_water = (field_capacity - wilting_point) * 1000

        # Estimate saturated hydraulic conductivity (Cosby et al., 1984)
        ksat = typical["ksat"]

        # Estimate CEC
        cec = self._estimate_cec(clay, organic_matter)

        return SoilProperties(
            sand=sand,
            silt=silt,
            clay=clay,
            texture_class=texture,
            bulk_density=bulk_density,
            porosity=porosity,
            field_capacity=field_capacity,
            wilting_point=wilting_point,
            available_water=available_water,
            saturated_conductivity=ksat,
            organic_matter=organic_matter,
            cec=cec,
        )

    def _estimate_cec(
        self,
        clay: float,
        organic_matter: float,
    ) -> float:
        """
        Estimate cation exchange capacity.

        Args:
            clay: Clay content (%)
            organic_matter: Organic matter (%)

        Returns:
            CEC (meq/100g)
        """
        # Simple model: CEC from clay and OM contributions
        cec_clay = clay * 0.5  # ~0.5 meq/100g per % clay
        cec_om = organic_matter * 2.0  # ~2 meq/100g per % OM

        return cec_clay + cec_om

    def infiltration_rate(
        self,
        properties: SoilProperties,
        initial_moisture: float = 0.2,
        time: float = 60.0,
    ) -> float:
        """
        Estimate infiltration rate using Green-Ampt equation.

        Args:
            properties: Soil properties
            initial_moisture: Initial volumetric water content
            time: Time since start of infiltration (minutes)

        Returns:
            Infiltration rate (mm/hour)
        """
        # Saturated hydraulic conductivity (mm/hour)
        ksat_mm_h = properties.saturated_conductivity * 10 / 24

        # Wetting front suction (mm)
        psi_f = 200  # Typical value

        # Initial deficit
        theta_d = properties.field_capacity - initial_moisture

        # Approximate infiltration rate
        if time <= 0:
            return ksat_mm_h * 10  # Very high initially

        # Simplified Green-Ampt
        f = ksat_mm_h * (1 + psi_f * theta_d / (ksat_mm_h * time / 60))

        return min(f, ksat_mm_h * 10)


class SoilClassifier:
    """
    Comprehensive soil classification system.

    Supports USDA soil taxonomy and WRB classification.
    """

    def classify_by_properties(
        self,
        properties: SoilProperties,
        drainage: str = "well",
        parent_material: str = "unknown",
    ) -> Dict[str, str]:
        """
        Classify soil based on properties.

        Args:
            properties: Soil properties
            drainage: Drainage class
            parent_material: Parent material

        Returns:
            Dict with classification information
        """
        classification = {
            "texture": properties.texture_class,
            "drainage": drainage,
            "fertility": self._assess_fertility(properties),
            "workability": self._assess_workability(properties),
        }

        # Simplified soil order classification
        if properties.clay > 40:
            classification["order"] = "Vertisols"
        elif properties.organic_matter > 10:
            classification["order"] = "Histosols"
        elif drainage == "poor":
            classification["order"] = "Gleysols"
        else:
            classification["order"] = "Cambisols"

        return classification

    def _assess_fertility(self, properties: SoilProperties) -> str:
        """Assess soil fertility."""
        score = 0
        score += min(properties.cec / 20, 1) * 33
        score += min(properties.organic_matter / 5, 1) * 33
        score += (1 - abs(properties.ph - 6.5) / 2) * 34

        if score > 70:
            return "high"
        elif score > 40:
            return "medium"
        else:
            return "low"

    def _assess_workability(self, properties: SoilProperties) -> str:
        """Assess soil workability."""
        if properties.clay > 40:
            return "difficult"
        elif properties.sand > 70:
            return "easy"
        else:
            return "moderate"
