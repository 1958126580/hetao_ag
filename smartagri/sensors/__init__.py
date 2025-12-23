"""
IoT Sensor Data Processing Module

Provides sensor data handling for smart agriculture:
- Multi-sensor data fusion
- Anomaly detection
- Calibration management
- Time series processing

Example:
    >>> from smartagri.sensors import SensorNetwork
    >>> network = SensorNetwork()
    >>> network.add_sensor('temp_1', 'temperature')
    >>> reading = network.get_reading('temp_1')
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "SensorNetwork",
    "SensorReading",
    "DataFusion",
    "CalibrationManager",
    "SensorType",
]


class SensorType(Enum):
    """Types of agricultural sensors."""
    TEMPERATURE = auto()
    HUMIDITY = auto()
    SOIL_MOISTURE = auto()
    SOIL_TEMP = auto()
    LIGHT = auto()
    CO2 = auto()
    PH = auto()
    EC = auto()
    RAIN = auto()
    WIND = auto()
    PRESSURE = auto()


@dataclass
class SensorReading:
    """
    Single sensor reading record.

    Attributes:
        sensor_id: Sensor identifier
        timestamp: Reading timestamp
        value: Measured value
        unit: Measurement unit
        quality: Data quality flag (0-1)
    """
    sensor_id: str
    timestamp: datetime
    value: float
    unit: str
    quality: float = 1.0


@dataclass
class Sensor:
    """Sensor device configuration."""
    sensor_id: str
    sensor_type: SensorType
    location: Tuple[float, float] = (0.0, 0.0)
    calibration_offset: float = 0.0
    calibration_scale: float = 1.0
    last_reading: Optional[SensorReading] = None


class SensorNetwork:
    """
    Agricultural sensor network management.

    Manages collection of sensors across the farm,
    handles data collection and quality control.

    Example:
        >>> network = SensorNetwork()
        >>> network.add_sensor('soil_1', SensorType.SOIL_MOISTURE)
        >>> network.record_reading('soil_1', 0.35)
    """

    def __init__(self):
        """Initialize sensor network."""
        self.sensors: Dict[str, Sensor] = {}
        self.readings: Dict[str, List[SensorReading]] = {}

    def add_sensor(
        self,
        sensor_id: str,
        sensor_type: SensorType,
        location: Tuple[float, float] = (0.0, 0.0),
    ) -> Sensor:
        """
        Add sensor to network.

        Args:
            sensor_id: Unique sensor identifier
            sensor_type: Type of sensor
            location: GPS coordinates

        Returns:
            Created Sensor object
        """
        sensor = Sensor(
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            location=location,
        )
        self.sensors[sensor_id] = sensor
        self.readings[sensor_id] = []
        return sensor

    def record_reading(
        self,
        sensor_id: str,
        value: float,
        timestamp: datetime = None,
        unit: str = None,
    ) -> SensorReading:
        """
        Record sensor reading.

        Args:
            sensor_id: Sensor identifier
            value: Measured value
            timestamp: Reading time (default: now)
            unit: Measurement unit

        Returns:
            Recorded SensorReading
        """
        if sensor_id not in self.sensors:
            raise ValueError(f"Unknown sensor: {sensor_id}")

        sensor = self.sensors[sensor_id]
        timestamp = timestamp or datetime.now()

        # Apply calibration
        calibrated_value = value * sensor.calibration_scale + sensor.calibration_offset

        # Determine unit
        if unit is None:
            unit = self._default_unit(sensor.sensor_type)

        reading = SensorReading(
            sensor_id=sensor_id,
            timestamp=timestamp,
            value=calibrated_value,
            unit=unit,
        )

        self.readings[sensor_id].append(reading)
        sensor.last_reading = reading

        return reading

    def get_reading(self, sensor_id: str) -> Optional[SensorReading]:
        """Get most recent reading for sensor."""
        return self.sensors.get(sensor_id, Sensor("", SensorType.TEMPERATURE)).last_reading

    def get_history(
        self,
        sensor_id: str,
        hours: int = 24,
    ) -> List[SensorReading]:
        """Get reading history for sensor."""
        if sensor_id not in self.readings:
            return []

        cutoff = datetime.now() - timedelta(hours=hours)
        return [r for r in self.readings[sensor_id] if r.timestamp >= cutoff]

    def _default_unit(self, sensor_type: SensorType) -> str:
        """Get default unit for sensor type."""
        units = {
            SensorType.TEMPERATURE: '°C',
            SensorType.HUMIDITY: '%',
            SensorType.SOIL_MOISTURE: 'm³/m³',
            SensorType.LIGHT: 'lux',
            SensorType.CO2: 'ppm',
            SensorType.PH: 'pH',
            SensorType.EC: 'dS/m',
            SensorType.RAIN: 'mm',
            SensorType.WIND: 'm/s',
            SensorType.PRESSURE: 'hPa',
        }
        return units.get(sensor_type, '')


class DataFusion:
    """
    Multi-sensor data fusion.

    Combines readings from multiple sensors for
    improved accuracy and spatial coverage.
    """

    def __init__(self):
        """Initialize data fusion system."""
        pass

    def weighted_average(
        self,
        readings: List[SensorReading],
        weights: List[float] = None,
    ) -> float:
        """
        Calculate weighted average of readings.

        Args:
            readings: List of sensor readings
            weights: Optional weights (default: equal)

        Returns:
            Fused value
        """
        if not readings:
            return 0.0

        values = np.array([r.value for r in readings])

        if weights is None:
            weights = np.ones(len(readings))
        else:
            weights = np.array(weights)

        # Weight by quality
        qualities = np.array([r.quality for r in readings])
        weights = weights * qualities

        if np.sum(weights) > 0:
            return np.average(values, weights=weights)
        return np.mean(values)

    def spatial_interpolation(
        self,
        sensors: List[Sensor],
        readings: List[SensorReading],
        target_location: Tuple[float, float],
    ) -> float:
        """
        Interpolate value at target location.

        Uses inverse distance weighting.
        """
        if not sensors or not readings:
            return 0.0

        # Calculate distances
        distances = []
        values = []

        for sensor, reading in zip(sensors, readings):
            dist = np.sqrt(
                (sensor.location[0] - target_location[0])**2 +
                (sensor.location[1] - target_location[1])**2
            )
            distances.append(dist)
            values.append(reading.value)

        distances = np.array(distances)
        values = np.array(values)

        # IDW
        if np.any(distances < 1e-10):
            return values[distances < 1e-10][0]

        weights = 1 / (distances ** 2)
        return np.sum(weights * values) / np.sum(weights)


class CalibrationManager:
    """
    Sensor calibration management.

    Handles calibration coefficients and
    drift detection.
    """

    def __init__(self):
        """Initialize calibration manager."""
        self.calibrations: Dict[str, Dict] = {}

    def calibrate(
        self,
        sensor_id: str,
        reference_values: np.ndarray,
        measured_values: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calibrate sensor using reference measurements.

        Args:
            sensor_id: Sensor to calibrate
            reference_values: True reference values
            measured_values: Sensor measurements

        Returns:
            Calibration coefficients
        """
        # Linear regression for calibration
        slope, intercept = np.polyfit(measured_values, reference_values, 1)

        self.calibrations[sensor_id] = {
            'scale': slope,
            'offset': intercept,
            'calibration_date': datetime.now(),
        }

        return {'scale': slope, 'offset': intercept}

    def apply_calibration(
        self,
        sensor_id: str,
        raw_value: float,
    ) -> float:
        """Apply calibration to raw reading."""
        if sensor_id not in self.calibrations:
            return raw_value

        cal = self.calibrations[sensor_id]
        return raw_value * cal['scale'] + cal['offset']

    def detect_drift(
        self,
        sensor_id: str,
        recent_readings: List[SensorReading],
        reference: float,
        threshold: float = 0.1,
    ) -> bool:
        """Detect calibration drift."""
        if not recent_readings:
            return False

        avg_value = np.mean([r.value for r in recent_readings])
        drift = abs(avg_value - reference) / reference if reference != 0 else 0

        return drift > threshold
