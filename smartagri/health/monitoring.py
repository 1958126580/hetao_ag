"""
Health Monitoring Module

Real-time animal health monitoring and alerting:
- Vital sign tracking
- Health scoring systems
- Anomaly detection
- Alert generation

Example:
    >>> monitor = HealthMonitor()
    >>> monitor.record_observation(animal_id, temperature=39.5, heart_rate=70)
    >>> status = monitor.get_health_status(animal_id)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Animal health status classification."""
    HEALTHY = "healthy"
    WATCH = "watch"  # Minor concerns
    ALERT = "alert"  # Needs attention
    CRITICAL = "critical"  # Immediate attention
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Health alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class VitalSigns:
    """
    Animal vital signs record.

    Attributes:
        temperature: Body temperature (°C)
        heart_rate: Heart rate (bpm)
        respiratory_rate: Respiratory rate (breaths/min)
        rumen_contractions: Rumen contractions per 2 min (cattle)
        rumination_time: Daily rumination time (minutes)
        activity_level: Activity score (0-100)
    """
    temperature: Optional[float] = None
    heart_rate: Optional[float] = None
    respiratory_rate: Optional[float] = None
    rumen_contractions: Optional[int] = None
    rumination_time: Optional[float] = None
    activity_level: Optional[float] = None


@dataclass
class HealthObservation:
    """
    Health observation record.

    Attributes:
        animal_id: Animal identifier
        timestamp: Observation timestamp
        vitals: Vital signs
        body_condition_score: BCS (1-9 for cattle)
        appetite: Appetite score (0-5)
        mobility: Mobility score (0-5)
        appearance: General appearance notes
        observer: Observer ID/name
    """
    animal_id: str
    timestamp: datetime
    vitals: VitalSigns = field(default_factory=VitalSigns)
    body_condition_score: Optional[float] = None
    appetite: Optional[int] = None
    mobility: Optional[int] = None
    appearance: str = ""
    observer: str = ""


@dataclass
class HealthAlert:
    """
    Health alert record.

    Attributes:
        animal_id: Animal identifier
        timestamp: Alert timestamp
        severity: Alert severity
        alert_type: Type of alert
        message: Alert message
        metrics: Triggering metrics
        recommended_action: Suggested action
        acknowledged: Whether alert acknowledged
    """
    animal_id: str
    timestamp: datetime
    severity: AlertSeverity
    alert_type: str
    message: str
    metrics: Dict[str, float] = field(default_factory=dict)
    recommended_action: str = ""
    acknowledged: bool = False


# Normal vital sign ranges by species
VITAL_RANGES = {
    "cattle": {
        "temperature": (38.0, 39.5),  # °C
        "heart_rate": (40, 80),  # bpm
        "respiratory_rate": (15, 30),  # breaths/min
        "rumen_contractions": (1, 2),  # per 2 min
        "rumination_time": (400, 600),  # min/day
    },
    "sheep": {
        "temperature": (38.5, 40.0),
        "heart_rate": (70, 90),
        "respiratory_rate": (15, 30),
    },
    "swine": {
        "temperature": (38.0, 39.5),
        "heart_rate": (60, 90),
        "respiratory_rate": (15, 25),
    },
    "goat": {
        "temperature": (38.5, 40.0),
        "heart_rate": (70, 90),
        "respiratory_rate": (15, 30),
    },
    "horse": {
        "temperature": (37.5, 38.5),
        "heart_rate": (28, 44),
        "respiratory_rate": (8, 16),
    },
}


class VitalSignAnalyzer:
    """
    Vital sign analysis and anomaly detection.

    Analyzes vital signs against normal ranges
    and detects concerning patterns.

    Example:
        >>> analyzer = VitalSignAnalyzer(species="cattle")
        >>> status = analyzer.analyze(vitals)
    """

    def __init__(self, species: str = "cattle"):
        """
        Initialize analyzer.

        Args:
            species: Animal species
        """
        self.species = species.lower()
        self.ranges = VITAL_RANGES.get(
            self.species,
            VITAL_RANGES["cattle"]
        )

    def analyze(
        self,
        vitals: VitalSigns,
    ) -> Dict[str, Any]:
        """
        Analyze vital signs.

        Args:
            vitals: VitalSigns record

        Returns:
            Dict with analysis results
        """
        results = {
            "status": HealthStatus.HEALTHY,
            "abnormalities": [],
            "scores": {},
        }

        # Check each vital sign
        checks = [
            ("temperature", vitals.temperature),
            ("heart_rate", vitals.heart_rate),
            ("respiratory_rate", vitals.respiratory_rate),
            ("rumen_contractions", vitals.rumen_contractions),
            ("rumination_time", vitals.rumination_time),
        ]

        for name, value in checks:
            if value is not None and name in self.ranges:
                low, high = self.ranges[name]
                score = self._calculate_score(value, low, high)
                results["scores"][name] = score

                if score < 0.5:
                    if value < low:
                        results["abnormalities"].append(
                            f"{name} low: {value} (normal: {low}-{high})"
                        )
                    else:
                        results["abnormalities"].append(
                            f"{name} high: {value} (normal: {low}-{high})"
                        )

        # Determine overall status
        if results["abnormalities"]:
            n_abnormal = len(results["abnormalities"])
            min_score = min(results["scores"].values()) if results["scores"] else 1

            if n_abnormal >= 3 or min_score < 0.2:
                results["status"] = HealthStatus.CRITICAL
            elif n_abnormal >= 2 or min_score < 0.4:
                results["status"] = HealthStatus.ALERT
            else:
                results["status"] = HealthStatus.WATCH

        return results

    def _calculate_score(
        self,
        value: float,
        low: float,
        high: float,
    ) -> float:
        """Calculate normalized score (0-1) for vital sign."""
        mid = (low + high) / 2
        half_range = (high - low) / 2

        # Distance from midpoint, normalized
        deviation = abs(value - mid) / half_range

        # Score drops as deviation increases
        score = max(0, 1 - (deviation - 1) ** 2) if deviation > 1 else 1

        return score

    def detect_fever(
        self,
        temperature: float,
    ) -> Dict[str, Any]:
        """
        Detect fever based on temperature.

        Args:
            temperature: Body temperature (°C)

        Returns:
            Dict with fever assessment
        """
        normal_high = self.ranges.get("temperature", (38, 39.5))[1]

        if temperature > normal_high + 1.5:
            return {
                "fever": True,
                "severity": "high",
                "temperature": temperature,
                "deviation": temperature - normal_high,
            }
        elif temperature > normal_high + 0.5:
            return {
                "fever": True,
                "severity": "moderate",
                "temperature": temperature,
                "deviation": temperature - normal_high,
            }
        elif temperature > normal_high:
            return {
                "fever": True,
                "severity": "mild",
                "temperature": temperature,
                "deviation": temperature - normal_high,
            }
        else:
            return {
                "fever": False,
                "temperature": temperature,
            }


class HealthScoring:
    """
    Health scoring system.

    Calculates composite health scores from
    multiple health indicators.

    Example:
        >>> scorer = HealthScoring()
        >>> score = scorer.calculate_score(observation)
    """

    # Component weights for health score
    WEIGHTS = {
        "vitals": 0.30,
        "body_condition": 0.20,
        "appetite": 0.15,
        "mobility": 0.15,
        "activity": 0.10,
        "appearance": 0.10,
    }

    def __init__(self, species: str = "cattle"):
        """
        Initialize health scoring.

        Args:
            species: Animal species
        """
        self.species = species.lower()
        self.vital_analyzer = VitalSignAnalyzer(species)

    def calculate_score(
        self,
        observation: HealthObservation,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive health score.

        Args:
            observation: HealthObservation record

        Returns:
            Dict with health scores
        """
        component_scores = {}

        # Vitals score
        if observation.vitals:
            vital_analysis = self.vital_analyzer.analyze(observation.vitals)
            vital_scores = list(vital_analysis["scores"].values())
            component_scores["vitals"] = np.mean(vital_scores) if vital_scores else 1.0

        # Body condition score (normalized to 0-1)
        if observation.body_condition_score is not None:
            # Optimal BCS is 5-7 for cattle (on 1-9 scale)
            bcs = observation.body_condition_score
            if 5 <= bcs <= 7:
                component_scores["body_condition"] = 1.0
            elif 4 <= bcs < 5 or 7 < bcs <= 8:
                component_scores["body_condition"] = 0.75
            else:
                component_scores["body_condition"] = 0.5

        # Appetite score (0-5 scale)
        if observation.appetite is not None:
            component_scores["appetite"] = observation.appetite / 5.0

        # Mobility score (0-5 scale)
        if observation.mobility is not None:
            component_scores["mobility"] = observation.mobility / 5.0

        # Activity level (0-100)
        if observation.vitals.activity_level is not None:
            component_scores["activity"] = observation.vitals.activity_level / 100.0

        # Calculate weighted score
        total_weight = 0
        weighted_sum = 0

        for component, score in component_scores.items():
            weight = self.WEIGHTS.get(component, 0.1)
            weighted_sum += score * weight
            total_weight += weight

        overall_score = weighted_sum / total_weight if total_weight > 0 else 0.5

        # Classify status
        if overall_score >= 0.85:
            status = HealthStatus.HEALTHY
        elif overall_score >= 0.65:
            status = HealthStatus.WATCH
        elif overall_score >= 0.45:
            status = HealthStatus.ALERT
        else:
            status = HealthStatus.CRITICAL

        return {
            "overall_score": overall_score,
            "component_scores": component_scores,
            "status": status,
            "timestamp": observation.timestamp,
        }


class HealthMonitor:
    """
    Real-time health monitoring system.

    Tracks health observations, generates alerts,
    and maintains health history for analysis.

    Example:
        >>> monitor = HealthMonitor(species="cattle")
        >>> monitor.record_observation(animal_id, temperature=39.5)
        >>> alerts = monitor.get_active_alerts()
    """

    def __init__(self, species: str = "cattle"):
        """
        Initialize health monitor.

        Args:
            species: Animal species
        """
        self.species = species.lower()
        self.scorer = HealthScoring(species)
        self.vital_analyzer = VitalSignAnalyzer(species)

        self._observations: Dict[str, List[HealthObservation]] = {}
        self._alerts: List[HealthAlert] = []
        self._status: Dict[str, HealthStatus] = {}

    def record_observation(
        self,
        animal_id: str,
        temperature: Optional[float] = None,
        heart_rate: Optional[float] = None,
        respiratory_rate: Optional[float] = None,
        rumen_contractions: Optional[int] = None,
        rumination_time: Optional[float] = None,
        activity_level: Optional[float] = None,
        body_condition_score: Optional[float] = None,
        appetite: Optional[int] = None,
        mobility: Optional[int] = None,
        appearance: str = "",
        observer: str = "",
        timestamp: Optional[datetime] = None,
    ) -> HealthObservation:
        """
        Record health observation.

        Args:
            animal_id: Animal identifier
            temperature: Body temperature (°C)
            heart_rate: Heart rate (bpm)
            respiratory_rate: Respiratory rate
            rumen_contractions: Rumen contractions
            rumination_time: Rumination time (min)
            activity_level: Activity level (0-100)
            body_condition_score: Body condition score
            appetite: Appetite score (0-5)
            mobility: Mobility score (0-5)
            appearance: General appearance notes
            observer: Observer name/ID
            timestamp: Observation timestamp

        Returns:
            HealthObservation record
        """
        if timestamp is None:
            timestamp = datetime.now()

        vitals = VitalSigns(
            temperature=temperature,
            heart_rate=heart_rate,
            respiratory_rate=respiratory_rate,
            rumen_contractions=rumen_contractions,
            rumination_time=rumination_time,
            activity_level=activity_level,
        )

        observation = HealthObservation(
            animal_id=animal_id,
            timestamp=timestamp,
            vitals=vitals,
            body_condition_score=body_condition_score,
            appetite=appetite,
            mobility=mobility,
            appearance=appearance,
            observer=observer,
        )

        # Store observation
        if animal_id not in self._observations:
            self._observations[animal_id] = []
        self._observations[animal_id].append(observation)

        # Update health status
        score_result = self.scorer.calculate_score(observation)
        self._status[animal_id] = score_result["status"]

        # Check for alerts
        self._check_alerts(animal_id, observation, score_result)

        return observation

    def _check_alerts(
        self,
        animal_id: str,
        observation: HealthObservation,
        score_result: Dict[str, Any],
    ) -> None:
        """Check for health alerts based on observation."""
        alerts_to_add = []

        # Check for fever
        if observation.vitals.temperature:
            fever = self.vital_analyzer.detect_fever(observation.vitals.temperature)
            if fever["fever"]:
                severity = {
                    "high": AlertSeverity.CRITICAL,
                    "moderate": AlertSeverity.HIGH,
                    "mild": AlertSeverity.MEDIUM,
                }.get(fever["severity"], AlertSeverity.LOW)

                alerts_to_add.append(HealthAlert(
                    animal_id=animal_id,
                    timestamp=observation.timestamp,
                    severity=severity,
                    alert_type="fever",
                    message=f"Fever detected: {fever['temperature']}°C",
                    metrics={"temperature": fever["temperature"]},
                    recommended_action="Isolate and monitor, consider veterinary exam",
                ))

        # Check for low appetite
        if observation.appetite is not None and observation.appetite <= 1:
            alerts_to_add.append(HealthAlert(
                animal_id=animal_id,
                timestamp=observation.timestamp,
                severity=AlertSeverity.HIGH,
                alert_type="appetite",
                message="Severely reduced appetite",
                metrics={"appetite": observation.appetite},
                recommended_action="Monitor closely, check for illness",
            ))

        # Check for mobility issues
        if observation.mobility is not None and observation.mobility <= 2:
            alerts_to_add.append(HealthAlert(
                animal_id=animal_id,
                timestamp=observation.timestamp,
                severity=AlertSeverity.MEDIUM,
                alert_type="mobility",
                message="Reduced mobility detected",
                metrics={"mobility": observation.mobility},
                recommended_action="Check for lameness, injury, or illness",
            ))

        # Critical overall status
        if score_result["status"] == HealthStatus.CRITICAL:
            alerts_to_add.append(HealthAlert(
                animal_id=animal_id,
                timestamp=observation.timestamp,
                severity=AlertSeverity.CRITICAL,
                alert_type="overall_health",
                message="Critical health status",
                metrics=score_result["component_scores"],
                recommended_action="Immediate veterinary attention required",
            ))

        self._alerts.extend(alerts_to_add)

    def get_health_status(
        self,
        animal_id: str,
    ) -> Dict[str, Any]:
        """
        Get current health status for animal.

        Args:
            animal_id: Animal identifier

        Returns:
            Dict with health status
        """
        observations = self._observations.get(animal_id, [])
        status = self._status.get(animal_id, HealthStatus.UNKNOWN)

        if not observations:
            return {
                "status": HealthStatus.UNKNOWN,
                "message": "No observations recorded",
            }

        latest = observations[-1]
        score_result = self.scorer.calculate_score(latest)

        return {
            "animal_id": animal_id,
            "status": status,
            "score": score_result["overall_score"],
            "last_observation": latest.timestamp,
            "component_scores": score_result["component_scores"],
        }

    def get_observations(
        self,
        animal_id: str,
        days: int = 30,
    ) -> List[HealthObservation]:
        """Get observation history for animal."""
        observations = self._observations.get(animal_id, [])
        cutoff = datetime.now() - timedelta(days=days)
        return [o for o in observations if o.timestamp >= cutoff]

    def get_active_alerts(
        self,
        animal_id: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
    ) -> List[HealthAlert]:
        """
        Get active (unacknowledged) alerts.

        Args:
            animal_id: Filter by animal
            severity: Filter by severity

        Returns:
            List of active alerts
        """
        alerts = [a for a in self._alerts if not a.acknowledged]

        if animal_id:
            alerts = [a for a in alerts if a.animal_id == animal_id]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts

    def acknowledge_alert(self, alert: HealthAlert) -> None:
        """Acknowledge an alert."""
        alert.acknowledged = True

    def get_health_trend(
        self,
        animal_id: str,
        days: int = 14,
    ) -> Dict[str, Any]:
        """
        Analyze health trend over time.

        Args:
            animal_id: Animal identifier
            days: Days to analyze

        Returns:
            Dict with trend analysis
        """
        observations = self.get_observations(animal_id, days)

        if len(observations) < 2:
            return {"status": "insufficient_data"}

        # Calculate scores over time
        scores = []
        temps = []
        for obs in observations:
            result = self.scorer.calculate_score(obs)
            scores.append(result["overall_score"])
            if obs.vitals.temperature:
                temps.append(obs.vitals.temperature)

        # Calculate trend (slope)
        x = np.arange(len(scores))
        if len(scores) >= 2:
            slope = np.polyfit(x, scores, 1)[0]
        else:
            slope = 0

        # Classify trend
        if slope > 0.01:
            trend = "improving"
        elif slope < -0.01:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "animal_id": animal_id,
            "n_observations": len(observations),
            "avg_score": np.mean(scores),
            "latest_score": scores[-1],
            "score_trend": trend,
            "slope": slope,
            "avg_temperature": np.mean(temps) if temps else None,
        }

    def get_herd_health_summary(self) -> Dict[str, Any]:
        """Get health summary for entire herd."""
        status_counts = {s: 0 for s in HealthStatus}
        scores = []

        for animal_id in self._observations:
            status = self._status.get(animal_id, HealthStatus.UNKNOWN)
            status_counts[status] += 1

            observations = self._observations[animal_id]
            if observations:
                result = self.scorer.calculate_score(observations[-1])
                scores.append(result["overall_score"])

        active_alerts = self.get_active_alerts()
        critical_alerts = [a for a in active_alerts
                          if a.severity == AlertSeverity.CRITICAL]

        return {
            "total_animals": len(self._observations),
            "status_distribution": {s.value: c for s, c in status_counts.items()},
            "avg_health_score": np.mean(scores) if scores else None,
            "active_alerts": len(active_alerts),
            "critical_alerts": len(critical_alerts),
            "animals_needing_attention": status_counts[HealthStatus.ALERT] +
                                         status_counts[HealthStatus.CRITICAL],
        }
