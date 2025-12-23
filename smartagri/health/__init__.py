"""
Animal Health Monitoring Module

Comprehensive animal health management and diagnostics:
- Disease detection and monitoring
- Health scoring and assessment
- Vaccination management
- Treatment tracking
- Biosecurity protocols

Example:
    >>> from smartagri.health import HealthMonitor, DiseaseDetector
    >>> monitor = HealthMonitor()
    >>> monitor.record_observation(animal_id, temperature=39.5)
    >>> alert = monitor.check_health_status(animal_id)
"""

from .monitoring import (
    HealthMonitor,
    HealthObservation,
    HealthAlert,
    VitalSigns,
    HealthStatus,
)
from .disease import (
    DiseaseDetector,
    DiseaseSymptom,
    DiseaseRisk,
    EpidemiologyTracker,
)
from .treatment import (
    TreatmentManager,
    TreatmentRecord,
    VaccinationScheduler,
    MedicationLog,
)
from .biosecurity import (
    BiosecurityManager,
    QuarantineZone,
    RiskAssessment,
)

__all__ = [
    # Monitoring
    "HealthMonitor",
    "HealthObservation",
    "HealthAlert",
    "VitalSigns",
    "HealthStatus",
    # Disease
    "DiseaseDetector",
    "DiseaseSymptom",
    "DiseaseRisk",
    "EpidemiologyTracker",
    # Treatment
    "TreatmentManager",
    "TreatmentRecord",
    "VaccinationScheduler",
    "MedicationLog",
    # Biosecurity
    "BiosecurityManager",
    "QuarantineZone",
    "RiskAssessment",
]
