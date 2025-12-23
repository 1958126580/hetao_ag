"""
Pest and Disease Detection Module

Provides computer vision-based pest and disease detection:
- Image classification for pest identification
- Disease symptom detection
- Severity assessment
- Treatment recommendations

Example:
    >>> from smartagri.pests import PestDetector
    >>> detector = PestDetector()
    >>> result = detector.identify(image)
"""

from smartagri.pests.detection import (
    PestDetector,
    DiseaseDetector,
    PestClassifier,
)

from smartagri.pests.management import (
    IPMPlanner,
    PesticideRecommender,
    ThresholdManager,
)

__all__ = [
    "PestDetector",
    "DiseaseDetector",
    "PestClassifier",
    "IPMPlanner",
    "PesticideRecommender",
    "ThresholdManager",
]
