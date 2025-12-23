"""
Integrated Pest Management Module

Provides pest management planning and decision support:
- Economic threshold calculations
- IPM strategy planning
- Pesticide recommendations
- Resistance management

Example:
    >>> planner = IPMPlanner()
    >>> strategy = planner.develop_strategy(pest_data, crop_info)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)


class TreatmentType(Enum):
    """Pest treatment types."""
    BIOLOGICAL = auto()
    CHEMICAL = auto()
    CULTURAL = auto()
    MECHANICAL = auto()
    GENETIC = auto()


@dataclass
class PesticideInfo:
    """
    Pesticide information record.

    Attributes:
        name: Product name
        active_ingredient: Active chemical
        mode_of_action: How it works
        target_pests: Pests controlled
        phi_days: Pre-harvest interval
        rei_hours: Restricted entry interval
        rate_per_ha: Application rate
    """
    name: str
    active_ingredient: str
    mode_of_action: str
    target_pests: List[str]
    phi_days: int
    rei_hours: int
    rate_per_ha: float


@dataclass
class IPMStrategy:
    """
    Integrated pest management strategy.

    Attributes:
        pest_target: Target pest
        threshold: Action threshold
        monitoring_interval: Days between scouting
        treatments: Prioritized treatment options
        rotation_plan: Resistance management rotation
    """
    pest_target: str
    threshold: float
    monitoring_interval: int
    treatments: List[Dict[str, Any]]
    rotation_plan: List[str]


class ThresholdManager:
    """
    Economic threshold calculator for pest management.

    Calculates action thresholds based on pest density,
    crop value, and control costs.

    Example:
        >>> manager = ThresholdManager()
        >>> threshold = manager.calculate_threshold(
        ...     pest_density=5,
        ...     crop_value=1000,
        ...     control_cost=50
        ... )
    """

    # Economic thresholds by pest (pests per plant or per sample)
    DEFAULT_THRESHOLDS = {
        'aphid': {'low': 50, 'moderate': 150, 'high': 500},
        'corn_borer': {'low': 0.5, 'moderate': 1.0, 'high': 2.0},
        'spider_mite': {'low': 5, 'moderate': 15, 'high': 30},
    }

    def calculate_threshold(
        self,
        pest_density: float,
        crop_value: float,
        control_cost: float,
        yield_loss_per_pest: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Calculate economic injury level and action threshold.

        Args:
            pest_density: Current pest density
            crop_value: Crop value per hectare ($)
            control_cost: Control cost per hectare ($)
            yield_loss_per_pest: Yield loss per pest unit

        Returns:
            Dictionary with threshold analysis

        Example:
            >>> result = manager.calculate_threshold(10, 2000, 50)
            >>> if result['action_needed']:
            ...     print("Treatment recommended")
        """
        # Economic Injury Level (EIL)
        # EIL = C / (V × I × D × K)
        # C = control cost, V = crop value, I = injury per pest
        # D = damage per unit injury, K = proportion kill

        damage_per_pest = crop_value * yield_loss_per_pest
        kill_efficacy = 0.85  # Assumed

        if damage_per_pest > 0:
            eil = control_cost / (damage_per_pest * kill_efficacy)
        else:
            eil = float('inf')

        # Economic Threshold (ET) = 80% of EIL
        et = eil * 0.8

        # Action determination
        action_needed = pest_density >= et

        return {
            'economic_injury_level': eil,
            'economic_threshold': et,
            'current_density': pest_density,
            'action_needed': action_needed,
            'urgency': 'high' if pest_density > eil else 'moderate' if action_needed else 'low',
            'potential_loss': pest_density * damage_per_pest,
            'benefit_cost_ratio': damage_per_pest * pest_density / control_cost if control_cost > 0 else 0,
        }

    def get_threshold(
        self,
        pest_name: str,
        severity: str = 'moderate',
    ) -> float:
        """Get standard threshold for known pest."""
        thresholds = self.DEFAULT_THRESHOLDS.get(pest_name, {})
        return thresholds.get(severity, 10.0)


class IPMPlanner:
    """
    Integrated Pest Management planning system.

    Develops comprehensive IPM strategies combining
    multiple control methods.

    Example:
        >>> planner = IPMPlanner()
        >>> strategy = planner.develop_strategy('aphid', 'corn')
        >>> for treatment in strategy.treatments:
        ...     print(f"{treatment['type']}: {treatment['method']}")
    """

    # IPM treatment options by pest
    TREATMENT_OPTIONS = {
        'aphid': [
            {'type': TreatmentType.BIOLOGICAL, 'method': 'release_ladybugs', 'efficacy': 0.7},
            {'type': TreatmentType.BIOLOGICAL, 'method': 'parasitic_wasps', 'efficacy': 0.65},
            {'type': TreatmentType.CULTURAL, 'method': 'remove_weeds', 'efficacy': 0.3},
            {'type': TreatmentType.CHEMICAL, 'method': 'insecticidal_soap', 'efficacy': 0.8},
            {'type': TreatmentType.CHEMICAL, 'method': 'neem_oil', 'efficacy': 0.75},
        ],
        'corn_borer': [
            {'type': TreatmentType.BIOLOGICAL, 'method': 'trichogramma_wasps', 'efficacy': 0.6},
            {'type': TreatmentType.BIOLOGICAL, 'method': 'bt_spray', 'efficacy': 0.85},
            {'type': TreatmentType.CULTURAL, 'method': 'crop_rotation', 'efficacy': 0.5},
            {'type': TreatmentType.GENETIC, 'method': 'bt_corn', 'efficacy': 0.95},
            {'type': TreatmentType.MECHANICAL, 'method': 'stalk_destruction', 'efficacy': 0.4},
        ],
    }

    def __init__(self):
        """Initialize IPM planner."""
        self.threshold_manager = ThresholdManager()

    def develop_strategy(
        self,
        pest_name: str,
        crop_type: str,
        pest_density: float = 0,
        organic_only: bool = False,
    ) -> IPMStrategy:
        """
        Develop IPM strategy for pest situation.

        Args:
            pest_name: Target pest
            crop_type: Crop being protected
            pest_density: Current pest density
            organic_only: Restrict to organic methods

        Returns:
            IPMStrategy with recommended actions
        """
        # Get treatment options
        treatments = self.TREATMENT_OPTIONS.get(pest_name, [])

        if organic_only:
            treatments = [t for t in treatments
                         if t['type'] in [TreatmentType.BIOLOGICAL, TreatmentType.CULTURAL,
                                         TreatmentType.MECHANICAL]]

        # Sort by efficacy
        treatments = sorted(treatments, key=lambda x: x['efficacy'], reverse=True)

        # Determine threshold
        threshold = self.threshold_manager.get_threshold(pest_name)

        # Create rotation plan for resistance management
        rotation_plan = self._create_rotation_plan(treatments)

        return IPMStrategy(
            pest_target=pest_name,
            threshold=threshold,
            monitoring_interval=7,  # Weekly
            treatments=[{
                'type': t['type'].name,
                'method': t['method'],
                'efficacy': t['efficacy'],
                'priority': i + 1,
            } for i, t in enumerate(treatments)],
            rotation_plan=rotation_plan,
        )

    def _create_rotation_plan(
        self,
        treatments: List[Dict],
    ) -> List[str]:
        """Create treatment rotation for resistance management."""
        # Group by type, rotate between groups
        by_type = {}
        for t in treatments:
            type_name = t['type'].name
            if type_name not in by_type:
                by_type[type_name] = []
            by_type[type_name].append(t['method'])

        rotation = []
        for type_name in ['BIOLOGICAL', 'CULTURAL', 'CHEMICAL']:
            if type_name in by_type and by_type[type_name]:
                rotation.append(by_type[type_name][0])

        return rotation


class PesticideRecommender:
    """
    Pesticide recommendation system.

    Provides pesticide recommendations considering
    efficacy, safety, and resistance management.

    Example:
        >>> recommender = PesticideRecommender()
        >>> options = recommender.recommend('aphid', 'corn')
    """

    # Pesticide database
    PESTICIDE_DB = {
        'imidacloprid': PesticideInfo(
            name='Imidacloprid',
            active_ingredient='Imidacloprid',
            mode_of_action='nicotinic_acetylcholine_receptor',
            target_pests=['aphid', 'whitefly', 'thrips'],
            phi_days=21,
            rei_hours=12,
            rate_per_ha=0.05,
        ),
        'spinosad': PesticideInfo(
            name='Spinosad',
            active_ingredient='Spinosyn A and D',
            mode_of_action='nicotinic_acetylcholine_receptor',
            target_pests=['corn_borer', 'thrips', 'leafminer'],
            phi_days=1,
            rei_hours=4,
            rate_per_ha=0.1,
        ),
        'bifenthrin': PesticideInfo(
            name='Bifenthrin',
            active_ingredient='Bifenthrin',
            mode_of_action='sodium_channel_modulator',
            target_pests=['spider_mite', 'aphid', 'corn_borer'],
            phi_days=14,
            rei_hours=12,
            rate_per_ha=0.2,
        ),
    }

    def recommend(
        self,
        pest_name: str,
        crop_type: str,
        days_to_harvest: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Recommend pesticides for pest situation.

        Args:
            pest_name: Target pest
            crop_type: Crop type
            days_to_harvest: Days until harvest

        Returns:
            List of recommended pesticides
        """
        recommendations = []

        for name, pesticide in self.PESTICIDE_DB.items():
            if pest_name in pesticide.target_pests:
                if pesticide.phi_days <= days_to_harvest:
                    recommendations.append({
                        'name': pesticide.name,
                        'active_ingredient': pesticide.active_ingredient,
                        'rate_per_ha': pesticide.rate_per_ha,
                        'phi_days': pesticide.phi_days,
                        'rei_hours': pesticide.rei_hours,
                        'mode_of_action': pesticide.mode_of_action,
                    })

        return recommendations

    def check_resistance_risk(
        self,
        pesticide_name: str,
        application_history: List[str],
    ) -> Dict[str, Any]:
        """
        Assess resistance development risk.

        Args:
            pesticide_name: Proposed pesticide
            application_history: Previous applications

        Returns:
            Resistance risk assessment
        """
        pesticide = self.PESTICIDE_DB.get(pesticide_name.lower())
        if not pesticide:
            return {'risk': 'unknown', 'reason': 'Pesticide not in database'}

        # Count same mode of action applications
        moa = pesticide.mode_of_action
        same_moa_count = sum(
            1 for p in application_history
            if p.lower() in self.PESTICIDE_DB and
            self.PESTICIDE_DB[p.lower()].mode_of_action == moa
        )

        if same_moa_count >= 3:
            risk = 'high'
            recommendation = 'Rotate to different mode of action'
        elif same_moa_count >= 2:
            risk = 'moderate'
            recommendation = 'Consider rotation after this application'
        else:
            risk = 'low'
            recommendation = 'Continue monitoring'

        return {
            'risk': risk,
            'same_moa_applications': same_moa_count,
            'recommendation': recommendation,
            'mode_of_action': moa,
        }
