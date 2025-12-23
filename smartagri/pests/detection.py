"""
Pest and Disease Detection Module

Computer vision-based detection of agricultural pests and diseases:
- Deep learning image classification
- Multi-pest detection
- Severity scoring
- Temporal tracking

Example:
    >>> detector = PestDetector(model='efficientnet')
    >>> result = detector.detect(image_path)
    >>> print(f"Detected: {result.pest_type} ({result.confidence:.2%})")
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)


class PestCategory(Enum):
    """Pest category classification."""
    INSECT = auto()
    MITE = auto()
    NEMATODE = auto()
    RODENT = auto()
    BIRD = auto()
    WEED = auto()
    UNKNOWN = auto()


class DiseaseSeverity(Enum):
    """Disease severity levels."""
    NONE = 0
    MILD = 1
    MODERATE = 2
    SEVERE = 3
    CRITICAL = 4


@dataclass
class DetectionResult:
    """
    Pest/disease detection result.

    Attributes:
        detected: Whether pest/disease was detected
        type_name: Name of detected pest/disease
        confidence: Detection confidence (0-1)
        category: Pest category
        severity: Severity level
        bounding_box: Location in image (x, y, w, h)
        recommendations: Treatment recommendations
    """
    detected: bool
    type_name: str
    confidence: float
    category: PestCategory = PestCategory.UNKNOWN
    severity: DiseaseSeverity = DiseaseSeverity.NONE
    bounding_box: Optional[Tuple[int, int, int, int]] = None
    recommendations: List[str] = field(default_factory=list)


class PestDetector:
    """
    Deep learning-based pest detection system.

    Uses convolutional neural networks to identify
    agricultural pests from images.

    Example:
        >>> detector = PestDetector()
        >>> result = detector.detect("field_image.jpg")
        >>> if result.detected:
        ...     print(f"Found {result.type_name}")
    """

    # Known pest types with characteristics
    PEST_DATABASE = {
        'aphid': {
            'category': PestCategory.INSECT,
            'damage_type': 'sap_sucking',
            'crops_affected': ['wheat', 'corn', 'soybean', 'vegetables'],
            'treatments': ['neem oil', 'insecticidal soap', 'ladybugs'],
        },
        'corn_borer': {
            'category': PestCategory.INSECT,
            'damage_type': 'boring',
            'crops_affected': ['corn', 'sorghum'],
            'treatments': ['Bt spray', 'beneficial wasps', 'crop rotation'],
        },
        'japanese_beetle': {
            'category': PestCategory.INSECT,
            'damage_type': 'defoliation',
            'crops_affected': ['soybean', 'corn', 'fruit_trees'],
            'treatments': ['hand picking', 'milky spore', 'neem'],
        },
        'spider_mite': {
            'category': PestCategory.MITE,
            'damage_type': 'sap_sucking',
            'crops_affected': ['vegetables', 'fruit_trees', 'cotton'],
            'treatments': ['miticides', 'predatory mites', 'water spray'],
        },
        'rootknot_nematode': {
            'category': PestCategory.NEMATODE,
            'damage_type': 'root_damage',
            'crops_affected': ['vegetables', 'cotton', 'soybean'],
            'treatments': ['crop rotation', 'resistant varieties', 'solarization'],
        },
    }

    def __init__(
        self,
        model_type: str = "cnn",
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize pest detector.

        Args:
            model_type: Model architecture ('cnn', 'efficientnet', 'resnet')
            confidence_threshold: Minimum confidence for detection
        """
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._is_loaded = False

    def load_model(self, model_path: Optional[str] = None) -> None:
        """Load detection model weights."""
        # Placeholder for model loading
        self._is_loaded = True
        logger.info(f"Loaded {self.model_type} pest detection model")

    def detect(
        self,
        image: Union[str, np.ndarray],
        return_all: bool = False,
    ) -> Union[DetectionResult, List[DetectionResult]]:
        """
        Detect pests in image.

        Args:
            image: Image path or numpy array
            return_all: Return all detections vs top only

        Returns:
            DetectionResult or list of DetectionResults

        Example:
            >>> result = detector.detect("crop_image.jpg")
            >>> print(f"Confidence: {result.confidence:.2%}")
        """
        # Simulate detection (in production, would use trained model)
        if isinstance(image, str):
            # Load image from path
            image_data = np.random.randn(224, 224, 3)  # Placeholder
        else:
            image_data = image

        # Simulate predictions
        pest_types = list(self.PEST_DATABASE.keys())
        confidences = np.random.dirichlet(np.ones(len(pest_types)))

        results = []
        for pest, conf in zip(pest_types, confidences):
            if conf > self.confidence_threshold:
                pest_info = self.PEST_DATABASE[pest]
                results.append(DetectionResult(
                    detected=True,
                    type_name=pest,
                    confidence=conf,
                    category=pest_info['category'],
                    severity=DiseaseSeverity.MODERATE,
                    recommendations=pest_info['treatments'],
                ))

        if not results:
            results = [DetectionResult(
                detected=False,
                type_name="none",
                confidence=0.0,
            )]

        if return_all:
            return sorted(results, key=lambda x: x.confidence, reverse=True)
        return max(results, key=lambda x: x.confidence)

    def detect_batch(
        self,
        images: List[Union[str, np.ndarray]],
    ) -> List[DetectionResult]:
        """Detect pests in multiple images."""
        return [self.detect(img) for img in images]


class DiseaseDetector:
    """
    Plant disease detection system.

    Identifies crop diseases from leaf and plant images
    using deep learning classification.

    Example:
        >>> detector = DiseaseDetector()
        >>> result = detector.diagnose("leaf_image.jpg")
        >>> print(f"Disease: {result.type_name}, Severity: {result.severity}")
    """

    DISEASE_DATABASE = {
        'powdery_mildew': {
            'symptoms': ['white_powder', 'curled_leaves', 'yellowing'],
            'crops': ['wheat', 'grapes', 'squash'],
            'treatments': ['fungicide', 'sulfur spray', 'improve_airflow'],
        },
        'rust': {
            'symptoms': ['orange_pustules', 'leaf_spots', 'defoliation'],
            'crops': ['wheat', 'corn', 'beans'],
            'treatments': ['fungicide', 'resistant_varieties', 'crop_rotation'],
        },
        'blight': {
            'symptoms': ['water_soaked_lesions', 'browning', 'wilting'],
            'crops': ['tomato', 'potato', 'peppers'],
            'treatments': ['copper_fungicide', 'remove_infected', 'spacing'],
        },
        'mosaic_virus': {
            'symptoms': ['mottled_leaves', 'distortion', 'stunting'],
            'crops': ['tobacco', 'tomato', 'cucumber'],
            'treatments': ['remove_infected', 'control_aphids', 'resistant_varieties'],
        },
        'root_rot': {
            'symptoms': ['wilting', 'yellowing', 'root_decay'],
            'crops': ['soybeans', 'cotton', 'vegetables'],
            'treatments': ['improve_drainage', 'fungicide_drench', 'rotation'],
        },
    }

    def __init__(
        self,
        confidence_threshold: float = 0.6,
    ):
        """Initialize disease detector."""
        self.confidence_threshold = confidence_threshold

    def diagnose(
        self,
        image: Union[str, np.ndarray],
        crop_type: Optional[str] = None,
    ) -> DetectionResult:
        """
        Diagnose plant disease from image.

        Args:
            image: Image path or array
            crop_type: Optional crop type for filtering

        Returns:
            DetectionResult with diagnosis
        """
        # Simulate diagnosis
        diseases = list(self.DISEASE_DATABASE.keys())

        if crop_type:
            # Filter to diseases affecting this crop
            diseases = [d for d in diseases
                       if crop_type in self.DISEASE_DATABASE[d]['crops']]

        if not diseases:
            diseases = list(self.DISEASE_DATABASE.keys())

        # Simulate prediction
        confidences = np.random.dirichlet(np.ones(len(diseases)))
        top_idx = np.argmax(confidences)
        top_disease = diseases[top_idx]
        top_conf = confidences[top_idx]

        if top_conf > self.confidence_threshold:
            disease_info = self.DISEASE_DATABASE[top_disease]
            return DetectionResult(
                detected=True,
                type_name=top_disease,
                confidence=top_conf,
                severity=self._assess_severity(top_conf),
                recommendations=disease_info['treatments'],
            )

        return DetectionResult(
            detected=False,
            type_name="healthy",
            confidence=1 - top_conf,
        )

    def _assess_severity(self, confidence: float) -> DiseaseSeverity:
        """Assess disease severity based on detection confidence."""
        if confidence > 0.9:
            return DiseaseSeverity.CRITICAL
        elif confidence > 0.8:
            return DiseaseSeverity.SEVERE
        elif confidence > 0.7:
            return DiseaseSeverity.MODERATE
        else:
            return DiseaseSeverity.MILD


class PestClassifier:
    """
    Multi-class pest classification.

    Classifies pests into taxonomic categories and
    species for precise identification.
    """

    def __init__(self):
        """Initialize pest classifier."""
        self.classes = list(PestDetector.PEST_DATABASE.keys())

    def classify(
        self,
        features: np.ndarray,
    ) -> Tuple[str, float]:
        """
        Classify pest from extracted features.

        Args:
            features: Feature vector from image

        Returns:
            Tuple of (class_name, probability)
        """
        # Simplified classification
        probs = np.random.dirichlet(np.ones(len(self.classes)))
        top_idx = np.argmax(probs)
        return self.classes[top_idx], probs[top_idx]

    def get_taxonomy(self, pest_name: str) -> Dict[str, str]:
        """Get taxonomic information for pest."""
        taxonomies = {
            'aphid': {'order': 'Hemiptera', 'family': 'Aphididae'},
            'corn_borer': {'order': 'Lepidoptera', 'family': 'Crambidae'},
            'japanese_beetle': {'order': 'Coleoptera', 'family': 'Scarabaeidae'},
        }
        return taxonomies.get(pest_name, {'order': 'Unknown', 'family': 'Unknown'})
