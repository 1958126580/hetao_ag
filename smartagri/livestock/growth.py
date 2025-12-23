"""
Livestock Growth Module

Models for animal growth prediction and weight management:
- Growth curve modeling (Gompertz, von Bertalanffy, logistic)
- Weight prediction and monitoring
- Feed conversion efficiency
- Body condition scoring

Example:
    >>> model = GrowthModel(species="cattle", breed="Angus")
    >>> predicted_weight = model.predict_weight(age_days=365, sex="male")
    >>> daily_gain = model.calculate_adg(weights=[200, 250, 300], days=[0, 30, 60])
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class GrowthCurveType(Enum):
    """Growth curve model types."""
    GOMPERTZ = "gompertz"
    VON_BERTALANFFY = "von_bertalanffy"
    LOGISTIC = "logistic"
    BRODY = "brody"
    RICHARDS = "richards"


@dataclass
class GrowthParameters:
    """
    Species/breed-specific growth parameters.

    Attributes:
        mature_weight: Asymptotic mature weight (kg)
        growth_rate: Growth rate parameter
        inflection_point: Age at maximum growth rate (days)
        birth_weight: Expected birth weight (kg)
        shape_parameter: Shape parameter for Richards model
    """
    mature_weight: float
    growth_rate: float
    inflection_point: float
    birth_weight: float
    shape_parameter: float = 1.0


# Default growth parameters by species and breed
GROWTH_PARAMETERS = {
    "cattle": {
        "angus": GrowthParameters(
            mature_weight=600, growth_rate=0.0035,
            inflection_point=250, birth_weight=35
        ),
        "hereford": GrowthParameters(
            mature_weight=580, growth_rate=0.0033,
            inflection_point=260, birth_weight=38
        ),
        "holstein": GrowthParameters(
            mature_weight=680, growth_rate=0.0030,
            inflection_point=280, birth_weight=42
        ),
        "brahman": GrowthParameters(
            mature_weight=550, growth_rate=0.0028,
            inflection_point=300, birth_weight=30
        ),
        "default": GrowthParameters(
            mature_weight=550, growth_rate=0.0032,
            inflection_point=270, birth_weight=36
        ),
    },
    "sheep": {
        "merino": GrowthParameters(
            mature_weight=70, growth_rate=0.008,
            inflection_point=90, birth_weight=4.5
        ),
        "suffolk": GrowthParameters(
            mature_weight=110, growth_rate=0.009,
            inflection_point=85, birth_weight=5.5
        ),
        "dorper": GrowthParameters(
            mature_weight=95, growth_rate=0.0085,
            inflection_point=88, birth_weight=4.0
        ),
        "default": GrowthParameters(
            mature_weight=80, growth_rate=0.0085,
            inflection_point=90, birth_weight=4.5
        ),
    },
    "swine": {
        "yorkshire": GrowthParameters(
            mature_weight=250, growth_rate=0.012,
            inflection_point=140, birth_weight=1.4
        ),
        "duroc": GrowthParameters(
            mature_weight=270, growth_rate=0.011,
            inflection_point=145, birth_weight=1.5
        ),
        "landrace": GrowthParameters(
            mature_weight=260, growth_rate=0.0115,
            inflection_point=142, birth_weight=1.4
        ),
        "default": GrowthParameters(
            mature_weight=250, growth_rate=0.012,
            inflection_point=140, birth_weight=1.4
        ),
    },
    "poultry": {
        "broiler": GrowthParameters(
            mature_weight=4.5, growth_rate=0.045,
            inflection_point=28, birth_weight=0.04
        ),
        "layer": GrowthParameters(
            mature_weight=2.2, growth_rate=0.030,
            inflection_point=42, birth_weight=0.04
        ),
        "turkey": GrowthParameters(
            mature_weight=15, growth_rate=0.025,
            inflection_point=70, birth_weight=0.06
        ),
        "default": GrowthParameters(
            mature_weight=3.5, growth_rate=0.040,
            inflection_point=35, birth_weight=0.04
        ),
    },
    "goat": {
        "boer": GrowthParameters(
            mature_weight=100, growth_rate=0.007,
            inflection_point=120, birth_weight=3.5
        ),
        "saanen": GrowthParameters(
            mature_weight=70, growth_rate=0.0065,
            inflection_point=130, birth_weight=3.0
        ),
        "default": GrowthParameters(
            mature_weight=75, growth_rate=0.007,
            inflection_point=125, birth_weight=3.2
        ),
    },
}


class GrowthCurve:
    """
    Mathematical growth curve models.

    Implements various nonlinear growth functions commonly
    used in animal science for predicting body weight.

    Example:
        >>> curve = GrowthCurve(GrowthCurveType.GOMPERTZ)
        >>> weight = curve.predict(age=365, params=params)
    """

    def __init__(self, curve_type: GrowthCurveType = GrowthCurveType.GOMPERTZ):
        """
        Initialize growth curve model.

        Args:
            curve_type: Type of growth curve model
        """
        self.curve_type = curve_type
        self._curve_functions = {
            GrowthCurveType.GOMPERTZ: self._gompertz,
            GrowthCurveType.VON_BERTALANFFY: self._von_bertalanffy,
            GrowthCurveType.LOGISTIC: self._logistic,
            GrowthCurveType.BRODY: self._brody,
            GrowthCurveType.RICHARDS: self._richards,
        }

    def predict(
        self,
        age: float,
        params: GrowthParameters,
    ) -> float:
        """
        Predict body weight at given age.

        Args:
            age: Age in days
            params: Growth parameters

        Returns:
            Predicted body weight (kg)
        """
        func = self._curve_functions[self.curve_type]
        return func(age, params)

    def predict_trajectory(
        self,
        ages: np.ndarray,
        params: GrowthParameters,
    ) -> np.ndarray:
        """
        Predict weight trajectory over multiple ages.

        Args:
            ages: Array of ages (days)
            params: Growth parameters

        Returns:
            Array of predicted weights
        """
        return np.array([self.predict(age, params) for age in ages])

    def _gompertz(self, t: float, params: GrowthParameters) -> float:
        """
        Gompertz growth function.

        W(t) = A * exp(-b * exp(-k * t))

        Where:
            A = mature weight
            b = integration constant (related to birth weight)
            k = growth rate
        """
        A = params.mature_weight
        k = params.growth_rate
        b = np.log(A / params.birth_weight)
        return A * np.exp(-b * np.exp(-k * t))

    def _von_bertalanffy(self, t: float, params: GrowthParameters) -> float:
        """
        Von Bertalanffy growth function.

        W(t) = A * (1 - b * exp(-k * t))^3
        """
        A = params.mature_weight
        k = params.growth_rate
        b = 1 - (params.birth_weight / A) ** (1/3)
        return A * (1 - b * np.exp(-k * t)) ** 3

    def _logistic(self, t: float, params: GrowthParameters) -> float:
        """
        Logistic growth function.

        W(t) = A / (1 + b * exp(-k * t))
        """
        A = params.mature_weight
        k = params.growth_rate
        b = (A - params.birth_weight) / params.birth_weight
        return A / (1 + b * np.exp(-k * t))

    def _brody(self, t: float, params: GrowthParameters) -> float:
        """
        Brody growth function.

        W(t) = A * (1 - b * exp(-k * t))
        """
        A = params.mature_weight
        k = params.growth_rate
        b = 1 - params.birth_weight / A
        return A * (1 - b * np.exp(-k * t))

    def _richards(self, t: float, params: GrowthParameters) -> float:
        """
        Richards (flexible) growth function.

        W(t) = A * (1 - b * exp(-k * t))^m

        Where m is the shape parameter.
        """
        A = params.mature_weight
        k = params.growth_rate
        m = params.shape_parameter
        b = 1 - (params.birth_weight / A) ** (1/m)
        return A * (1 - b * np.exp(-k * t)) ** m

    def daily_gain(
        self,
        age: float,
        params: GrowthParameters,
    ) -> float:
        """
        Calculate instantaneous daily gain at given age.

        Args:
            age: Age in days
            params: Growth parameters

        Returns:
            Daily weight gain (kg/day)
        """
        # Numerical derivative
        dt = 0.5
        w1 = self.predict(age - dt, params)
        w2 = self.predict(age + dt, params)
        return (w2 - w1) / (2 * dt)


class GrowthModel:
    """
    Comprehensive livestock growth modeling.

    Provides weight prediction, growth analysis, and
    performance monitoring for livestock operations.

    Example:
        >>> model = GrowthModel(species="cattle", breed="angus")
        >>> weight = model.predict_weight(age_days=365)
        >>> adg = model.calculate_adg([200, 300], [0, 60])
    """

    def __init__(
        self,
        species: str,
        breed: str = "default",
        curve_type: GrowthCurveType = GrowthCurveType.GOMPERTZ,
        custom_params: Optional[GrowthParameters] = None,
    ):
        """
        Initialize growth model.

        Args:
            species: Animal species
            breed: Breed name
            curve_type: Growth curve model type
            custom_params: Custom growth parameters
        """
        self.species = species.lower()
        self.breed = breed.lower()
        self.curve = GrowthCurve(curve_type)

        if custom_params:
            self.params = custom_params
        else:
            self.params = self._get_default_params()

    def _get_default_params(self) -> GrowthParameters:
        """Get default parameters for species/breed."""
        species_params = GROWTH_PARAMETERS.get(self.species, {})
        params = species_params.get(self.breed)

        if params is None:
            params = species_params.get("default")

        if params is None:
            # Ultimate fallback
            logger.warning(
                f"No parameters for {self.species}/{self.breed}, "
                "using cattle defaults"
            )
            params = GROWTH_PARAMETERS["cattle"]["default"]

        return params

    def predict_weight(
        self,
        age_days: int,
        sex: str = "unknown",
        adjustment_factor: float = 1.0,
    ) -> float:
        """
        Predict animal weight at given age.

        Args:
            age_days: Age in days
            sex: Animal sex (male/female/unknown)
            adjustment_factor: Manual adjustment factor

        Returns:
            Predicted weight (kg)
        """
        base_weight = self.curve.predict(age_days, self.params)

        # Sex adjustment (males typically 10-15% heavier)
        sex_factor = 1.0
        if sex.lower() == "male":
            sex_factor = 1.10
        elif sex.lower() == "female":
            sex_factor = 0.95

        return base_weight * sex_factor * adjustment_factor

    def predict_weight_range(
        self,
        age_days: int,
        confidence: float = 0.95,
    ) -> Tuple[float, float, float]:
        """
        Predict weight with confidence interval.

        Args:
            age_days: Age in days
            confidence: Confidence level

        Returns:
            Tuple of (mean, lower_bound, upper_bound)
        """
        mean_weight = self.predict_weight(age_days)

        # Estimate CV based on age (young animals more variable)
        cv = 0.12 + 0.05 * np.exp(-age_days / 365)
        std = mean_weight * cv

        # Z-score for confidence interval
        z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)

        lower = mean_weight - z * std
        upper = mean_weight + z * std

        return mean_weight, max(0, lower), upper

    def calculate_adg(
        self,
        weights: List[float],
        days: List[int],
    ) -> float:
        """
        Calculate average daily gain from weight records.

        Args:
            weights: List of recorded weights (kg)
            days: List of days since start

        Returns:
            Average daily gain (kg/day)
        """
        if len(weights) < 2 or len(days) < 2:
            return 0.0

        weights = np.array(weights)
        days = np.array(days)

        # Linear regression for ADG
        n = len(weights)
        sum_x = np.sum(days)
        sum_y = np.sum(weights)
        sum_xy = np.sum(days * weights)
        sum_x2 = np.sum(days ** 2)

        adg = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)

        return adg

    def days_to_target_weight(
        self,
        current_weight: float,
        target_weight: float,
        current_age_days: int,
    ) -> int:
        """
        Estimate days required to reach target weight.

        Args:
            current_weight: Current weight (kg)
            target_weight: Target weight (kg)
            current_age_days: Current age in days

        Returns:
            Estimated days to target
        """
        if target_weight <= current_weight:
            return 0

        # Search for age where predicted weight meets target
        for days_ahead in range(1, 1000):
            future_age = current_age_days + days_ahead
            predicted = self.predict_weight(future_age)
            if predicted >= target_weight:
                return days_ahead

        return 999  # Target may not be achievable

    def growth_efficiency(
        self,
        weight_gain: float,
        feed_consumed: float,
    ) -> Dict[str, float]:
        """
        Calculate growth efficiency metrics.

        Args:
            weight_gain: Total weight gain (kg)
            feed_consumed: Total feed consumed (kg)

        Returns:
            Dict with efficiency metrics
        """
        if feed_consumed <= 0:
            return {"fcr": 0, "feed_efficiency": 0, "g_f": 0}

        # Feed Conversion Ratio (kg feed per kg gain)
        fcr = feed_consumed / weight_gain if weight_gain > 0 else float('inf')

        # Feed Efficiency (kg gain per kg feed)
        feed_efficiency = weight_gain / feed_consumed

        # Gain to Feed ratio (same as feed efficiency, different name)
        g_f = feed_efficiency

        return {
            "fcr": fcr,
            "feed_efficiency": feed_efficiency,
            "g_f": g_f,
        }


class BodyConditionScore:
    """
    Body condition scoring system.

    Assesses animal body condition using standardized
    scoring scales for different species.

    Example:
        >>> bcs = BodyConditionScore(species="cattle")
        >>> score = bcs.estimate_from_weight(weight=500, frame_score=5)
    """

    # BCS scales by species (min, max, ideal_range)
    BCS_SCALES = {
        "cattle": {"min": 1, "max": 9, "ideal": (5, 7)},
        "sheep": {"min": 1, "max": 5, "ideal": (2.5, 3.5)},
        "swine": {"min": 1, "max": 5, "ideal": (3, 4)},
        "goat": {"min": 1, "max": 5, "ideal": (2.5, 3.5)},
        "horse": {"min": 1, "max": 9, "ideal": (5, 6)},
    }

    def __init__(self, species: str):
        """
        Initialize body condition scoring.

        Args:
            species: Animal species
        """
        self.species = species.lower()
        self.scale = self.BCS_SCALES.get(
            self.species,
            {"min": 1, "max": 5, "ideal": (2.5, 3.5)}
        )

    def estimate_from_weight(
        self,
        weight: float,
        frame_score: float,
        species_params: Optional[GrowthParameters] = None,
    ) -> float:
        """
        Estimate BCS from weight and frame size.

        Args:
            weight: Current weight (kg)
            frame_score: Frame size score (1-10)
            species_params: Growth parameters

        Returns:
            Estimated body condition score
        """
        if species_params is None:
            species_params = GROWTH_PARAMETERS.get(
                self.species, {}
            ).get("default")

        if species_params is None:
            return (self.scale["min"] + self.scale["max"]) / 2

        # Adjust mature weight for frame
        adjusted_mature = species_params.mature_weight * (0.8 + 0.04 * frame_score)

        # Weight ratio to adjusted mature
        weight_ratio = weight / adjusted_mature

        # Map to BCS scale
        scale_range = self.scale["max"] - self.scale["min"]
        mid_scale = (self.scale["min"] + self.scale["max"]) / 2

        # Sigmoid-like mapping centered at 0.85 of mature weight
        bcs = mid_scale + scale_range * 0.4 * np.tanh(3 * (weight_ratio - 0.85))

        # Clamp to valid range
        return np.clip(bcs, self.scale["min"], self.scale["max"])

    def is_ideal(self, score: float) -> bool:
        """Check if score is in ideal range."""
        return self.scale["ideal"][0] <= score <= self.scale["ideal"][1]

    def recommendation(self, score: float) -> str:
        """Get management recommendation based on BCS."""
        ideal_min, ideal_max = self.scale["ideal"]

        if score < ideal_min:
            deficit = ideal_min - score
            if deficit > (self.scale["max"] - self.scale["min"]) * 0.3:
                return "severely_underconditioned"
            return "underconditioned"
        elif score > ideal_max:
            excess = score - ideal_max
            if excess > (self.scale["max"] - self.scale["min"]) * 0.3:
                return "severely_overconditioned"
            return "overconditioned"
        else:
            return "optimal"


class WeightMonitor:
    """
    Animal weight monitoring and analysis.

    Tracks weight records and provides growth analysis
    and performance alerts.

    Example:
        >>> monitor = WeightMonitor(species="cattle")
        >>> monitor.add_record(animal_id, date, weight)
        >>> analysis = monitor.analyze_growth(animal_id)
    """

    def __init__(self, species: str, breed: str = "default"):
        """
        Initialize weight monitor.

        Args:
            species: Animal species
            breed: Breed name
        """
        self.species = species
        self.breed = breed
        self.growth_model = GrowthModel(species, breed)
        self._records: Dict[str, List[Tuple[date, float]]] = {}

    def add_record(
        self,
        animal_id: str,
        record_date: date,
        weight: float,
    ) -> None:
        """
        Add weight record for animal.

        Args:
            animal_id: Animal identifier
            record_date: Date of weighing
            weight: Weight in kg
        """
        if animal_id not in self._records:
            self._records[animal_id] = []

        self._records[animal_id].append((record_date, weight))

        # Keep sorted by date
        self._records[animal_id].sort(key=lambda x: x[0])

    def get_records(
        self,
        animal_id: str,
    ) -> List[Tuple[date, float]]:
        """Get all weight records for animal."""
        return self._records.get(animal_id, [])

    def analyze_growth(
        self,
        animal_id: str,
        birth_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """
        Analyze growth performance.

        Args:
            animal_id: Animal identifier
            birth_date: Animal birth date

        Returns:
            Dict with growth analysis
        """
        records = self.get_records(animal_id)

        if len(records) < 2:
            return {"status": "insufficient_data"}

        dates, weights = zip(*records)
        dates = list(dates)
        weights = list(weights)

        # Calculate days from first record
        days = [(d - dates[0]).days for d in dates]

        # Average daily gain
        adg = self.growth_model.calculate_adg(weights, days)

        # Current vs expected
        current_weight = weights[-1]
        if birth_date:
            age_days = (dates[-1] - birth_date).days
            expected = self.growth_model.predict_weight(age_days)
            weight_ratio = current_weight / expected
        else:
            age_days = None
            expected = None
            weight_ratio = None

        # Growth trend (recent vs overall)
        if len(weights) >= 4:
            recent_adg = self.growth_model.calculate_adg(
                weights[-4:], days[-4:]
            )
            trend = "accelerating" if recent_adg > adg * 1.1 else \
                    "decelerating" if recent_adg < adg * 0.9 else "stable"
        else:
            recent_adg = adg
            trend = "unknown"

        return {
            "animal_id": animal_id,
            "n_records": len(records),
            "first_weight": weights[0],
            "last_weight": current_weight,
            "total_gain": current_weight - weights[0],
            "adg": adg,
            "recent_adg": recent_adg,
            "trend": trend,
            "age_days": age_days,
            "expected_weight": expected,
            "weight_ratio": weight_ratio,
            "performance": self._classify_performance(weight_ratio),
        }

    def _classify_performance(
        self,
        weight_ratio: Optional[float],
    ) -> str:
        """Classify growth performance."""
        if weight_ratio is None:
            return "unknown"

        if weight_ratio >= 1.1:
            return "excellent"
        elif weight_ratio >= 0.95:
            return "good"
        elif weight_ratio >= 0.85:
            return "fair"
        else:
            return "poor"

    def detect_anomalies(
        self,
        animal_id: str,
        threshold_std: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """
        Detect weight anomalies (sudden changes).

        Args:
            animal_id: Animal identifier
            threshold_std: Standard deviation threshold

        Returns:
            List of anomaly records
        """
        records = self.get_records(animal_id)

        if len(records) < 3:
            return []

        dates, weights = zip(*records)
        weights = np.array(weights)

        # Calculate weight changes
        changes = np.diff(weights)
        mean_change = np.mean(changes)
        std_change = np.std(changes)

        anomalies = []
        for i, change in enumerate(changes):
            if std_change > 0:
                z_score = abs(change - mean_change) / std_change
                if z_score > threshold_std:
                    anomalies.append({
                        "date": dates[i + 1],
                        "weight": weights[i + 1],
                        "change": change,
                        "z_score": z_score,
                        "type": "gain" if change > mean_change else "loss",
                    })

        return anomalies


class FeedConversionAnalyzer:
    """
    Feed conversion and efficiency analysis.

    Analyzes feed intake relative to weight gain
    for optimizing feeding strategies.

    Example:
        >>> analyzer = FeedConversionAnalyzer()
        >>> analyzer.add_intake(animal_id, date, feed_kg)
        >>> fcr = analyzer.calculate_fcr(animal_id, start_weight, end_weight)
    """

    def __init__(self):
        """Initialize feed conversion analyzer."""
        self._intake_records: Dict[str, List[Tuple[date, float]]] = {}

    def add_intake(
        self,
        animal_id: str,
        record_date: date,
        feed_kg: float,
        feed_type: str = "standard",
    ) -> None:
        """
        Record feed intake.

        Args:
            animal_id: Animal identifier
            record_date: Date of feeding
            feed_kg: Amount of feed (kg)
            feed_type: Type of feed
        """
        if animal_id not in self._intake_records:
            self._intake_records[animal_id] = []

        self._intake_records[animal_id].append((record_date, feed_kg))

    def get_total_intake(
        self,
        animal_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> float:
        """
        Get total feed intake for period.

        Args:
            animal_id: Animal identifier
            start_date: Period start
            end_date: Period end

        Returns:
            Total feed intake (kg)
        """
        records = self._intake_records.get(animal_id, [])

        total = 0.0
        for record_date, amount in records:
            if start_date and record_date < start_date:
                continue
            if end_date and record_date > end_date:
                continue
            total += amount

        return total

    def calculate_fcr(
        self,
        animal_id: str,
        start_weight: float,
        end_weight: float,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> Dict[str, float]:
        """
        Calculate feed conversion ratio.

        Args:
            animal_id: Animal identifier
            start_weight: Starting weight (kg)
            end_weight: Ending weight (kg)
            start_date: Period start
            end_date: Period end

        Returns:
            Dict with FCR metrics
        """
        total_intake = self.get_total_intake(animal_id, start_date, end_date)
        weight_gain = end_weight - start_weight

        if weight_gain <= 0:
            return {
                "fcr": float('inf'),
                "feed_efficiency": 0,
                "total_intake": total_intake,
                "weight_gain": weight_gain,
            }

        fcr = total_intake / weight_gain
        feed_efficiency = weight_gain / total_intake if total_intake > 0 else 0

        return {
            "fcr": fcr,
            "feed_efficiency": feed_efficiency,
            "total_intake": total_intake,
            "weight_gain": weight_gain,
        }

    def benchmark_fcr(
        self,
        species: str,
        fcr: float,
    ) -> str:
        """
        Benchmark FCR against industry standards.

        Args:
            species: Animal species
            fcr: Calculated FCR

        Returns:
            Performance classification
        """
        # Industry benchmark FCR ranges
        benchmarks = {
            "cattle": {"excellent": 5.0, "good": 6.5, "average": 7.5},
            "swine": {"excellent": 2.5, "good": 3.0, "average": 3.5},
            "poultry": {"excellent": 1.6, "good": 1.8, "average": 2.0},
            "sheep": {"excellent": 5.0, "good": 6.0, "average": 7.0},
        }

        thresholds = benchmarks.get(species.lower(), benchmarks["cattle"])

        if fcr <= thresholds["excellent"]:
            return "excellent"
        elif fcr <= thresholds["good"]:
            return "good"
        elif fcr <= thresholds["average"]:
            return "average"
        else:
            return "below_average"
