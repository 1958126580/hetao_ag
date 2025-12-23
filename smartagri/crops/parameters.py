"""
Crop Parameters Database

Comprehensive database of crop-specific parameters for growth modeling,
yield prediction, and management decision support.

Contains parameters for major crops including:
- Cereals (corn, wheat, rice, barley)
- Legumes (soybean, beans, peas)
- Oil crops (canola, sunflower)
- Fiber crops (cotton)
- Vegetables and fruits
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class CropParameters:
    """
    Comprehensive crop parameter set.

    Attributes:
        name: Crop name
        scientific_name: Scientific name
        category: Crop category (cereal, legume, etc.)
        growth: Growth model parameters
        phenology: Phenological thresholds
        nutrition: Nutrient requirements
        water: Water requirements
        stress: Stress response parameters
        harvest: Harvest parameters
    """
    name: str
    scientific_name: str
    category: str

    # Temperature parameters (°C)
    t_base: float = 10.0  # Base temperature for GDD
    t_opt_min: float = 20.0  # Optimal range minimum
    t_opt_max: float = 30.0  # Optimal range maximum
    t_max: float = 40.0  # Maximum temperature

    # Phenology (GDD thresholds)
    gdd_emergence: float = 80.0
    gdd_flowering: float = 800.0
    gdd_maturity: float = 1400.0

    # Growth parameters
    rue: float = 3.0  # Radiation use efficiency (g/MJ)
    extinction_coef: float = 0.6  # Light extinction coefficient
    max_lai: float = 6.0  # Maximum LAI
    sla: float = 0.02  # Specific leaf area (m2/g)
    harvest_index: float = 0.45  # Harvest index

    # Root parameters
    max_root_depth: float = 1.5  # Maximum root depth (m)
    root_growth_rate: float = 0.02  # Root growth rate (m/day)

    # Water requirements
    kc_initial: float = 0.3  # Initial crop coefficient
    kc_mid: float = 1.15  # Mid-season crop coefficient
    kc_end: float = 0.5  # End-season crop coefficient
    water_stress_threshold: float = 0.5  # Soil water threshold

    # Nutrient requirements (kg/t yield)
    n_content: float = 25.0  # Nitrogen
    p_content: float = 5.0  # Phosphorus
    k_content: float = 20.0  # Potassium

    # Yield parameters
    potential_yield: float = 15000.0  # Potential yield (kg/ha)
    harvest_moisture: float = 15.0  # Target harvest moisture (%)

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


# Comprehensive crop database
CROP_DATABASE: Dict[str, CropParameters] = {
    "corn": CropParameters(
        name="Corn",
        scientific_name="Zea mays",
        category="cereal",
        t_base=10.0,
        t_opt_min=25.0,
        t_opt_max=33.0,
        t_max=40.0,
        gdd_emergence=80.0,
        gdd_flowering=850.0,
        gdd_maturity=1500.0,
        rue=3.8,
        extinction_coef=0.65,
        max_lai=6.0,
        harvest_index=0.48,
        max_root_depth=2.0,
        kc_initial=0.3,
        kc_mid=1.2,
        kc_end=0.35,
        n_content=25.0,
        p_content=4.0,
        k_content=20.0,
        potential_yield=15000.0,
        harvest_moisture=15.0,
        metadata={
            "photosynthesis": "C4",
            "growth_habit": "determinate",
            "pollination": "wind",
        },
    ),
    "wheat_winter": CropParameters(
        name="Winter Wheat",
        scientific_name="Triticum aestivum",
        category="cereal",
        t_base=0.0,
        t_opt_min=15.0,
        t_opt_max=22.0,
        t_max=32.0,
        gdd_emergence=100.0,
        gdd_flowering=1100.0,
        gdd_maturity=1800.0,
        rue=2.8,
        extinction_coef=0.55,
        max_lai=5.0,
        harvest_index=0.42,
        max_root_depth=1.5,
        kc_initial=0.4,
        kc_mid=1.15,
        kc_end=0.25,
        n_content=28.0,
        p_content=5.0,
        k_content=18.0,
        potential_yield=10000.0,
        harvest_moisture=13.0,
        metadata={
            "photosynthesis": "C3",
            "vernalization_required": True,
            "growth_habit": "determinate",
        },
    ),
    "wheat_spring": CropParameters(
        name="Spring Wheat",
        scientific_name="Triticum aestivum",
        category="cereal",
        t_base=0.0,
        t_opt_min=15.0,
        t_opt_max=24.0,
        t_max=32.0,
        gdd_emergence=80.0,
        gdd_flowering=800.0,
        gdd_maturity=1400.0,
        rue=2.6,
        extinction_coef=0.55,
        max_lai=4.5,
        harvest_index=0.40,
        max_root_depth=1.2,
        potential_yield=8000.0,
        harvest_moisture=13.0,
    ),
    "soybean": CropParameters(
        name="Soybean",
        scientific_name="Glycine max",
        category="legume",
        t_base=10.0,
        t_opt_min=25.0,
        t_opt_max=30.0,
        t_max=38.0,
        gdd_emergence=90.0,
        gdd_flowering=700.0,
        gdd_maturity=1400.0,
        rue=2.5,
        extinction_coef=0.7,
        max_lai=5.5,
        harvest_index=0.35,
        max_root_depth=1.5,
        kc_initial=0.4,
        kc_mid=1.15,
        kc_end=0.5,
        n_content=65.0,  # High due to protein content
        p_content=6.0,
        k_content=25.0,
        potential_yield=5000.0,
        harvest_moisture=13.0,
        metadata={
            "nitrogen_fixation": True,
            "photoperiod_sensitive": True,
        },
    ),
    "rice": CropParameters(
        name="Rice",
        scientific_name="Oryza sativa",
        category="cereal",
        t_base=12.0,
        t_opt_min=25.0,
        t_opt_max=32.0,
        t_max=40.0,
        gdd_emergence=80.0,
        gdd_flowering=900.0,
        gdd_maturity=1600.0,
        rue=2.6,
        extinction_coef=0.6,
        max_lai=6.0,
        harvest_index=0.45,
        max_root_depth=0.6,
        kc_initial=1.1,  # Flooded conditions
        kc_mid=1.2,
        kc_end=0.9,
        n_content=20.0,
        p_content=4.0,
        k_content=25.0,
        potential_yield=12000.0,
        harvest_moisture=22.0,
        metadata={
            "water_regime": "flooded",
            "photosynthesis": "C3",
        },
    ),
    "cotton": CropParameters(
        name="Cotton",
        scientific_name="Gossypium hirsutum",
        category="fiber",
        t_base=15.0,
        t_opt_min=28.0,
        t_opt_max=35.0,
        t_max=42.0,
        gdd_emergence=100.0,
        gdd_flowering=1000.0,
        gdd_maturity=1800.0,
        rue=1.8,
        extinction_coef=0.7,
        max_lai=4.0,
        harvest_index=0.35,
        max_root_depth=1.8,
        kc_initial=0.35,
        kc_mid=1.2,
        kc_end=0.7,
        n_content=30.0,
        p_content=5.0,
        k_content=15.0,
        potential_yield=5000.0,  # Lint + seed
        harvest_moisture=8.0,
        metadata={
            "growth_habit": "indeterminate",
            "fruiting_branches": True,
        },
    ),
    "potato": CropParameters(
        name="Potato",
        scientific_name="Solanum tuberosum",
        category="tuber",
        t_base=7.0,
        t_opt_min=15.0,
        t_opt_max=22.0,
        t_max=30.0,
        gdd_emergence=200.0,
        gdd_flowering=600.0,
        gdd_maturity=1200.0,
        rue=3.0,
        extinction_coef=0.6,
        max_lai=4.0,
        harvest_index=0.75,  # High for tuber crops
        max_root_depth=0.6,
        kc_initial=0.5,
        kc_mid=1.15,
        kc_end=0.75,
        n_content=4.0,  # Per ton fresh weight
        p_content=0.6,
        k_content=5.5,
        potential_yield=50000.0,  # Fresh weight
        harvest_moisture=80.0,  # Fresh tuber moisture
    ),
    "sunflower": CropParameters(
        name="Sunflower",
        scientific_name="Helianthus annuus",
        category="oilseed",
        t_base=8.0,
        t_opt_min=25.0,
        t_opt_max=30.0,
        t_max=38.0,
        gdd_emergence=100.0,
        gdd_flowering=800.0,
        gdd_maturity=1400.0,
        rue=2.2,
        extinction_coef=0.8,
        max_lai=4.5,
        harvest_index=0.30,
        max_root_depth=2.0,
        kc_initial=0.35,
        kc_mid=1.1,
        kc_end=0.35,
        n_content=40.0,
        p_content=7.0,
        k_content=30.0,
        potential_yield=4000.0,
        harvest_moisture=10.0,
    ),
    "barley": CropParameters(
        name="Barley",
        scientific_name="Hordeum vulgare",
        category="cereal",
        t_base=0.0,
        t_opt_min=15.0,
        t_opt_max=22.0,
        t_max=30.0,
        gdd_emergence=80.0,
        gdd_flowering=700.0,
        gdd_maturity=1200.0,
        rue=2.5,
        extinction_coef=0.55,
        max_lai=4.5,
        harvest_index=0.45,
        max_root_depth=1.2,
        potential_yield=8000.0,
        harvest_moisture=12.5,
    ),
    "canola": CropParameters(
        name="Canola",
        scientific_name="Brassica napus",
        category="oilseed",
        t_base=5.0,
        t_opt_min=18.0,
        t_opt_max=25.0,
        t_max=32.0,
        gdd_emergence=100.0,
        gdd_flowering=900.0,
        gdd_maturity=1500.0,
        rue=2.0,
        extinction_coef=0.7,
        max_lai=5.0,
        harvest_index=0.25,
        max_root_depth=1.5,
        n_content=50.0,
        p_content=8.0,
        k_content=20.0,
        potential_yield=4500.0,
        harvest_moisture=9.0,
    ),
    "alfalfa": CropParameters(
        name="Alfalfa",
        scientific_name="Medicago sativa",
        category="forage",
        t_base=5.0,
        t_opt_min=18.0,
        t_opt_max=28.0,
        t_max=35.0,
        gdd_emergence=150.0,
        gdd_flowering=500.0,  # Per cutting
        gdd_maturity=700.0,
        rue=2.2,
        extinction_coef=0.8,
        max_lai=5.0,
        harvest_index=0.90,  # Harvest whole plant
        max_root_depth=3.0,  # Deep tap root
        n_content=35.0,
        p_content=3.0,
        k_content=25.0,
        potential_yield=18000.0,  # Dry matter
        harvest_moisture=15.0,
        metadata={
            "nitrogen_fixation": True,
            "perennial": True,
            "cuttings_per_year": 4,
        },
    ),
}


def get_crop_parameters(crop_name: str) -> Optional[CropParameters]:
    """
    Get parameters for a specific crop.

    Args:
        crop_name: Crop name (case-insensitive)

    Returns:
        CropParameters or None if not found

    Example:
        >>> params = get_crop_parameters("corn")
        >>> print(f"Corn base temp: {params.t_base}°C")
    """
    # Normalize name
    crop_key = crop_name.lower().replace(" ", "_")

    # Direct lookup
    if crop_key in CROP_DATABASE:
        return CROP_DATABASE[crop_key]

    # Alias lookup
    aliases = {
        "maize": "corn",
        "soybeans": "soybean",
        "soya": "soybean",
        "rapeseed": "canola",
        "winter_wheat": "wheat_winter",
        "spring_wheat": "wheat_spring",
        "lucerne": "alfalfa",
    }

    if crop_key in aliases:
        return CROP_DATABASE.get(aliases[crop_key])

    logger.warning(f"Crop '{crop_name}' not found in database")
    return None


def list_available_crops() -> List[str]:
    """
    List all available crop names.

    Returns:
        List of crop names
    """
    return list(CROP_DATABASE.keys())


def get_crops_by_category(category: str) -> List[CropParameters]:
    """
    Get all crops in a category.

    Args:
        category: Crop category (cereal, legume, oilseed, etc.)

    Returns:
        List of CropParameters in that category
    """
    return [
        crop for crop in CROP_DATABASE.values()
        if crop.category.lower() == category.lower()
    ]
