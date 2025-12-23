"""
Disease Detection Module

Disease identification and epidemiological tracking:
- Symptom-based disease identification
- Risk assessment
- Outbreak detection
- Epidemiological modeling

Example:
    >>> detector = DiseaseDetector(species="cattle")
    >>> symptoms = ["fever", "nasal_discharge", "cough"]
    >>> diagnosis = detector.identify_disease(symptoms)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DiseaseCategory(Enum):
    """Disease category classification."""
    RESPIRATORY = "respiratory"
    DIGESTIVE = "digestive"
    REPRODUCTIVE = "reproductive"
    METABOLIC = "metabolic"
    INFECTIOUS = "infectious"
    PARASITIC = "parasitic"
    MUSCULOSKELETAL = "musculoskeletal"
    SKIN = "skin"
    NEUROLOGICAL = "neurological"
    OTHER = "other"


class TransmissionMode(Enum):
    """Disease transmission modes."""
    DIRECT_CONTACT = "direct_contact"
    AIRBORNE = "airborne"
    VECTOR = "vector"
    FECAL_ORAL = "fecal_oral"
    VERTICAL = "vertical"
    FOMITE = "fomite"
    NOT_TRANSMISSIBLE = "not_transmissible"


@dataclass
class DiseaseSymptom:
    """
    Disease symptom description.

    Attributes:
        name: Symptom name
        severity: Severity (mild, moderate, severe)
        description: Detailed description
        diagnostic_weight: Weight in diagnosis (0-1)
    """
    name: str
    severity: str = "moderate"
    description: str = ""
    diagnostic_weight: float = 1.0


@dataclass
class Disease:
    """
    Disease definition.

    Attributes:
        name: Disease name
        category: Disease category
        symptoms: Associated symptoms
        incubation_days: Incubation period
        transmission: Transmission mode
        mortality_rate: Mortality rate (0-1)
        zoonotic: Whether transmissible to humans
        notifiable: Whether reportable to authorities
    """
    name: str
    category: DiseaseCategory
    symptoms: List[str]
    incubation_days: Tuple[int, int] = (3, 7)
    transmission: TransmissionMode = TransmissionMode.NOT_TRANSMISSIBLE
    mortality_rate: float = 0.0
    zoonotic: bool = False
    notifiable: bool = False
    treatment: str = ""
    prevention: str = ""


@dataclass
class DiseaseRisk:
    """
    Disease risk assessment.

    Attributes:
        disease_name: Disease name
        probability: Risk probability (0-1)
        matching_symptoms: Symptoms that match
        missing_symptoms: Expected symptoms not present
        confidence: Assessment confidence
    """
    disease_name: str
    probability: float
    matching_symptoms: List[str]
    missing_symptoms: List[str]
    confidence: float


# Common livestock diseases database
DISEASE_DATABASE = {
    "cattle": [
        Disease(
            name="Bovine Respiratory Disease (BRD)",
            category=DiseaseCategory.RESPIRATORY,
            symptoms=["fever", "nasal_discharge", "cough", "rapid_breathing",
                      "depression", "reduced_appetite"],
            incubation_days=(3, 10),
            transmission=TransmissionMode.AIRBORNE,
            mortality_rate=0.05,
            treatment="Antibiotics, anti-inflammatories",
            prevention="Vaccination, stress reduction",
        ),
        Disease(
            name="Bovine Viral Diarrhea (BVD)",
            category=DiseaseCategory.INFECTIOUS,
            symptoms=["fever", "diarrhea", "nasal_discharge", "oral_lesions",
                      "reduced_milk", "abortion"],
            incubation_days=(5, 10),
            transmission=TransmissionMode.DIRECT_CONTACT,
            mortality_rate=0.10,
            notifiable=True,
            treatment="Supportive care",
            prevention="Vaccination, PI animal removal",
        ),
        Disease(
            name="Mastitis",
            category=DiseaseCategory.INFECTIOUS,
            symptoms=["swollen_udder", "abnormal_milk", "fever", "reduced_milk",
                      "pain", "depression"],
            incubation_days=(1, 3),
            transmission=TransmissionMode.FOMITE,
            mortality_rate=0.01,
            treatment="Antibiotics, milking management",
            prevention="Hygiene, teat dipping",
        ),
        Disease(
            name="Foot Rot",
            category=DiseaseCategory.MUSCULOSKELETAL,
            symptoms=["lameness", "swelling_foot", "foul_odor", "fever",
                      "reduced_appetite"],
            incubation_days=(1, 7),
            transmission=TransmissionMode.FECAL_ORAL,
            mortality_rate=0.0,
            treatment="Antibiotics, foot bath",
            prevention="Foot baths, dry conditions",
        ),
        Disease(
            name="Bloat",
            category=DiseaseCategory.DIGESTIVE,
            symptoms=["distended_abdomen", "difficulty_breathing", "reluctance_move",
                      "salivation", "kicking_belly"],
            incubation_days=(0, 1),
            transmission=TransmissionMode.NOT_TRANSMISSIBLE,
            mortality_rate=0.15,
            treatment="Trocar, anti-bloat agents",
            prevention="Gradual diet changes, ionophores",
        ),
        Disease(
            name="Milk Fever (Hypocalcemia)",
            category=DiseaseCategory.METABOLIC,
            symptoms=["weakness", "recumbency", "cold_ears", "dilated_pupils",
                      "reduced_appetite", "muscle_tremors"],
            incubation_days=(0, 2),
            transmission=TransmissionMode.NOT_TRANSMISSIBLE,
            mortality_rate=0.05,
            treatment="IV calcium",
            prevention="Prepartum diet management",
        ),
        Disease(
            name="Ketosis",
            category=DiseaseCategory.METABOLIC,
            symptoms=["reduced_appetite", "weight_loss", "reduced_milk",
                      "acetone_breath", "depression"],
            incubation_days=(7, 21),
            transmission=TransmissionMode.NOT_TRANSMISSIBLE,
            mortality_rate=0.01,
            treatment="Glucose, propylene glycol",
            prevention="Nutrition management",
        ),
        Disease(
            name="Anaplasmosis",
            category=DiseaseCategory.PARASITIC,
            symptoms=["fever", "anemia", "jaundice", "weakness", "aggression",
                      "reduced_appetite"],
            incubation_days=(15, 45),
            transmission=TransmissionMode.VECTOR,
            mortality_rate=0.30,
            treatment="Tetracycline",
            prevention="Tick control, vaccination",
        ),
    ],
    "sheep": [
        Disease(
            name="Footrot",
            category=DiseaseCategory.MUSCULOSKELETAL,
            symptoms=["lameness", "swelling_foot", "foul_odor", "separation"],
            incubation_days=(3, 14),
            transmission=TransmissionMode.FECAL_ORAL,
            mortality_rate=0.0,
            treatment="Foot trimming, antibiotics",
            prevention="Foot baths, culling carriers",
        ),
        Disease(
            name="Pneumonia",
            category=DiseaseCategory.RESPIRATORY,
            symptoms=["fever", "cough", "nasal_discharge", "rapid_breathing",
                      "depression"],
            incubation_days=(2, 7),
            transmission=TransmissionMode.AIRBORNE,
            mortality_rate=0.10,
            treatment="Antibiotics",
            prevention="Good ventilation, reduce stress",
        ),
        Disease(
            name="Pregnancy Toxemia",
            category=DiseaseCategory.METABOLIC,
            symptoms=["separation", "blindness", "head_pressing", "grinding_teeth",
                      "recumbency"],
            incubation_days=(14, 28),
            transmission=TransmissionMode.NOT_TRANSMISSIBLE,
            mortality_rate=0.70,
            treatment="IV glucose, propylene glycol",
            prevention="Late pregnancy nutrition",
        ),
    ],
    "swine": [
        Disease(
            name="Porcine Reproductive and Respiratory Syndrome (PRRS)",
            category=DiseaseCategory.INFECTIOUS,
            symptoms=["fever", "respiratory_distress", "abortion", "stillbirths",
                      "weak_piglets", "reduced_appetite"],
            incubation_days=(4, 14),
            transmission=TransmissionMode.AIRBORNE,
            mortality_rate=0.10,
            notifiable=True,
            treatment="Supportive care",
            prevention="Vaccination, biosecurity",
        ),
        Disease(
            name="Swine Dysentery",
            category=DiseaseCategory.DIGESTIVE,
            symptoms=["bloody_diarrhea", "mucus_feces", "weight_loss", "dehydration"],
            incubation_days=(7, 14),
            transmission=TransmissionMode.FECAL_ORAL,
            mortality_rate=0.10,
            treatment="Antibiotics",
            prevention="All-in all-out, biosecurity",
        ),
    ],
}


class DiseaseDetector:
    """
    Disease identification from symptoms.

    Uses symptom matching and Bayesian probability
    to identify likely diseases.

    Example:
        >>> detector = DiseaseDetector(species="cattle")
        >>> symptoms = ["fever", "cough", "nasal_discharge"]
        >>> results = detector.identify_disease(symptoms)
    """

    def __init__(self, species: str = "cattle"):
        """
        Initialize disease detector.

        Args:
            species: Animal species
        """
        self.species = species.lower()
        self.diseases = DISEASE_DATABASE.get(
            self.species,
            DISEASE_DATABASE["cattle"]
        )

        # Build symptom index
        self._symptom_index: Dict[str, List[str]] = {}
        for disease in self.diseases:
            for symptom in disease.symptoms:
                if symptom not in self._symptom_index:
                    self._symptom_index[symptom] = []
                self._symptom_index[symptom].append(disease.name)

    def identify_disease(
        self,
        symptoms: List[str],
        severity_weights: Optional[Dict[str, float]] = None,
    ) -> List[DiseaseRisk]:
        """
        Identify possible diseases from symptoms.

        Args:
            symptoms: Observed symptoms
            severity_weights: Symptom severity weights

        Returns:
            List of DiseaseRisk ranked by probability
        """
        symptoms_set = set(s.lower().replace(" ", "_") for s in symptoms)
        results = []

        for disease in self.diseases:
            disease_symptoms = set(disease.symptoms)

            # Calculate matches
            matching = symptoms_set & disease_symptoms
            missing = disease_symptoms - symptoms_set

            if not matching:
                continue

            # Calculate probability
            # Base on percentage of disease symptoms matched
            match_ratio = len(matching) / len(disease_symptoms)

            # Adjust for symptoms that don't fit
            extra_symptoms = symptoms_set - disease_symptoms
            penalty = len(extra_symptoms) * 0.1

            probability = max(0, match_ratio - penalty)

            # Confidence based on number of symptoms
            confidence = min(1.0, len(matching) / 3)

            results.append(DiseaseRisk(
                disease_name=disease.name,
                probability=probability,
                matching_symptoms=list(matching),
                missing_symptoms=list(missing),
                confidence=confidence,
            ))

        # Sort by probability
        results.sort(key=lambda x: x.probability, reverse=True)

        return results

    def get_disease_info(
        self,
        disease_name: str,
    ) -> Optional[Disease]:
        """Get detailed disease information."""
        for disease in self.diseases:
            if disease.name.lower() == disease_name.lower():
                return disease
        return None

    def get_differential_diagnosis(
        self,
        symptoms: List[str],
        n_top: int = 5,
    ) -> Dict[str, Any]:
        """
        Generate differential diagnosis report.

        Args:
            symptoms: Observed symptoms
            n_top: Number of top diagnoses

        Returns:
            Dict with differential diagnosis
        """
        risks = self.identify_disease(symptoms)[:n_top]

        differentials = []
        for risk in risks:
            disease = self.get_disease_info(risk.disease_name)
            differentials.append({
                "disease": risk.disease_name,
                "probability": risk.probability,
                "confidence": risk.confidence,
                "matching_symptoms": risk.matching_symptoms,
                "additional_tests": self._suggest_tests(disease),
                "category": disease.category.value if disease else None,
            })

        return {
            "observed_symptoms": symptoms,
            "differentials": differentials,
            "recommended_action": self._recommend_action(risks),
        }

    def _suggest_tests(
        self,
        disease: Optional[Disease],
    ) -> List[str]:
        """Suggest diagnostic tests for disease."""
        if disease is None:
            return ["General health panel"]

        tests = []
        if disease.category == DiseaseCategory.RESPIRATORY:
            tests = ["Nasal swab PCR", "Blood gas analysis", "X-ray"]
        elif disease.category == DiseaseCategory.DIGESTIVE:
            tests = ["Fecal analysis", "Blood chemistry", "Rumen fluid analysis"]
        elif disease.category == DiseaseCategory.INFECTIOUS:
            tests = ["Serology", "PCR", "Culture"]
        elif disease.category == DiseaseCategory.METABOLIC:
            tests = ["Blood glucose", "Blood ketones", "Calcium levels"]
        else:
            tests = ["Complete blood count", "Blood chemistry"]

        return tests

    def _recommend_action(
        self,
        risks: List[DiseaseRisk],
    ) -> str:
        """Recommend action based on risk assessment."""
        if not risks:
            return "Monitor closely, collect more observations"

        top_risk = risks[0]

        if top_risk.probability > 0.7:
            return f"High likelihood of {top_risk.disease_name}. Consult veterinarian immediately."
        elif top_risk.probability > 0.4:
            return f"Possible {top_risk.disease_name}. Further diagnostic tests recommended."
        else:
            return "No definitive diagnosis. Continue monitoring and collect more symptoms."


class EpidemiologyTracker:
    """
    Epidemiological tracking and outbreak detection.

    Monitors disease cases across the herd and
    detects potential outbreaks.

    Example:
        >>> tracker = EpidemiologyTracker()
        >>> tracker.record_case(animal_id, disease, date)
        >>> outbreak = tracker.detect_outbreak()
    """

    def __init__(self):
        """Initialize epidemiology tracker."""
        self._cases: List[Dict[str, Any]] = []
        self._baseline_rates: Dict[str, float] = {}

    def record_case(
        self,
        animal_id: str,
        disease: str,
        diagnosis_date: date,
        location: Optional[str] = None,
        confirmed: bool = False,
    ) -> None:
        """
        Record disease case.

        Args:
            animal_id: Animal identifier
            disease: Disease name
            diagnosis_date: Date of diagnosis
            location: Location/pen
            confirmed: Whether confirmed by vet
        """
        self._cases.append({
            "animal_id": animal_id,
            "disease": disease,
            "date": diagnosis_date,
            "location": location,
            "confirmed": confirmed,
            "recorded_at": datetime.now(),
        })

    def set_baseline_rate(
        self,
        disease: str,
        rate: float,
    ) -> None:
        """Set expected baseline rate for disease (cases per 1000 per week)."""
        self._baseline_rates[disease] = rate

    def get_case_count(
        self,
        disease: Optional[str] = None,
        days: int = 7,
        location: Optional[str] = None,
    ) -> int:
        """Get case count for period."""
        cutoff = date.today() - timedelta(days=days)

        cases = self._cases
        if disease:
            cases = [c for c in cases if c["disease"] == disease]
        if location:
            cases = [c for c in cases if c["location"] == location]

        return sum(1 for c in cases if c["date"] >= cutoff)

    def calculate_incidence(
        self,
        disease: str,
        herd_size: int,
        days: int = 7,
    ) -> float:
        """
        Calculate disease incidence rate.

        Args:
            disease: Disease name
            herd_size: Total herd size
            days: Time period

        Returns:
            Incidence rate (per 1000 animals per week)
        """
        cases = self.get_case_count(disease, days)
        weekly_rate = cases / (days / 7)
        return weekly_rate / herd_size * 1000

    def detect_outbreak(
        self,
        herd_size: int,
        threshold_factor: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """
        Detect disease outbreaks.

        Args:
            herd_size: Total herd size
            threshold_factor: Factor above baseline for outbreak

        Returns:
            List of detected outbreaks
        """
        outbreaks = []

        # Get unique diseases in recent cases
        recent = date.today() - timedelta(days=14)
        recent_diseases = set(
            c["disease"] for c in self._cases
            if c["date"] >= recent
        )

        for disease in recent_diseases:
            current_rate = self.calculate_incidence(disease, herd_size, 7)
            baseline = self._baseline_rates.get(disease, 1.0)

            if current_rate > baseline * threshold_factor:
                # Potential outbreak
                cases_7d = self.get_case_count(disease, 7)
                cases_14d = self.get_case_count(disease, 14)

                outbreaks.append({
                    "disease": disease,
                    "current_rate": current_rate,
                    "baseline_rate": baseline,
                    "ratio": current_rate / baseline,
                    "cases_7d": cases_7d,
                    "cases_14d": cases_14d,
                    "trend": "increasing" if cases_7d > cases_14d / 2 else "stable",
                    "severity": "high" if current_rate > baseline * 3 else "moderate",
                })

        return outbreaks

    def get_spatial_cluster(
        self,
        disease: str,
        days: int = 7,
    ) -> Dict[str, int]:
        """
        Identify spatial clustering of cases.

        Args:
            disease: Disease name
            days: Time period

        Returns:
            Dict of location -> case count
        """
        cutoff = date.today() - timedelta(days=days)
        cases = [
            c for c in self._cases
            if c["disease"] == disease and c["date"] >= cutoff
        ]

        cluster = {}
        for case in cases:
            loc = case.get("location", "unknown")
            cluster[loc] = cluster.get(loc, 0) + 1

        return cluster

    def generate_report(
        self,
        herd_size: int,
    ) -> Dict[str, Any]:
        """Generate epidemiology report."""
        # Cases by disease
        disease_counts = {}
        for case in self._cases:
            disease = case["disease"]
            disease_counts[disease] = disease_counts.get(disease, 0) + 1

        # Weekly trend
        weeks = []
        for w in range(4):
            start = date.today() - timedelta(days=7 * (w + 1))
            end = date.today() - timedelta(days=7 * w)
            count = sum(
                1 for c in self._cases
                if start <= c["date"] < end
            )
            weeks.append({"week": w + 1, "cases": count})

        return {
            "total_cases": len(self._cases),
            "by_disease": disease_counts,
            "weekly_trend": weeks,
            "active_outbreaks": self.detect_outbreak(herd_size),
            "herd_size": herd_size,
        }
