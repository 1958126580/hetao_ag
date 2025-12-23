"""
Data Validation Utilities for SmartAgri

Provides comprehensive validation functions for agricultural data,
ensuring data quality and type safety throughout the library.

Features:
    - Array shape and type validation
    - Range and bounds checking
    - Coordinate system validation
    - Date and time range validation
    - Missing data detection
    - Outlier detection for sensor data
"""

import numpy as np
from typing import (
    Optional,
    Union,
    List,
    Tuple,
    Any,
    TypeVar,
    Callable,
)
from datetime import datetime, date
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, List, Tuple]
T = TypeVar("T")


class ValidationError(Exception):
    """Exception raised for validation failures."""

    pass


@dataclass
class ValidationResult:
    """
    Container for validation results.

    Attributes:
        is_valid: Overall validity status
        errors: List of error messages
        warnings: List of warning messages
        stats: Statistics about validated data
    """

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Optional[dict] = None


def validate_array(
    arr: ArrayLike,
    name: str = "array",
    dtype: Optional[type] = None,
    ndim: Optional[int] = None,
    shape: Optional[Tuple[int, ...]] = None,
    min_size: int = 0,
    allow_nan: bool = False,
    allow_inf: bool = False,
) -> np.ndarray:
    """
    Validate and convert input to numpy array.

    Args:
        arr: Input array-like object
        name: Name for error messages
        dtype: Required data type
        ndim: Required number of dimensions
        shape: Required shape (use -1 for any size)
        min_size: Minimum number of elements
        allow_nan: Allow NaN values
        allow_inf: Allow infinite values

    Returns:
        Validated numpy array

    Raises:
        ValidationError: If validation fails

    Example:
        >>> data = validate_array([1, 2, 3], "yields", dtype=float, min_size=1)
    """
    # Convert to numpy array
    try:
        arr = np.asarray(arr)
    except Exception as e:
        raise ValidationError(f"{name}: Cannot convert to array - {e}")

    # Check dtype
    if dtype is not None:
        try:
            arr = arr.astype(dtype)
        except Exception as e:
            raise ValidationError(f"{name}: Cannot convert to {dtype} - {e}")

    # Check dimensions
    if ndim is not None and arr.ndim != ndim:
        raise ValidationError(
            f"{name}: Expected {ndim} dimensions, got {arr.ndim}"
        )

    # Check shape
    if shape is not None:
        for i, (expected, actual) in enumerate(zip(shape, arr.shape)):
            if expected != -1 and expected != actual:
                raise ValidationError(
                    f"{name}: Dimension {i} expected {expected}, got {actual}"
                )

    # Check size
    if arr.size < min_size:
        raise ValidationError(
            f"{name}: Expected at least {min_size} elements, got {arr.size}"
        )

    # Check for NaN
    if not allow_nan and np.any(np.isnan(arr)):
        n_nan = np.sum(np.isnan(arr))
        raise ValidationError(f"{name}: Contains {n_nan} NaN values")

    # Check for inf
    if not allow_inf and np.any(np.isinf(arr)):
        n_inf = np.sum(np.isinf(arr))
        raise ValidationError(f"{name}: Contains {n_inf} infinite values")

    return arr


def validate_positive(
    value: Union[int, float, np.ndarray],
    name: str = "value",
    strict: bool = False,
) -> Union[int, float, np.ndarray]:
    """
    Validate that value is positive.

    Args:
        value: Value to validate
        name: Name for error messages
        strict: If True, must be > 0; if False, >= 0

    Returns:
        Validated value

    Raises:
        ValidationError: If validation fails
    """
    if isinstance(value, np.ndarray):
        if strict:
            if np.any(value <= 0):
                raise ValidationError(f"{name}: All values must be > 0")
        else:
            if np.any(value < 0):
                raise ValidationError(f"{name}: All values must be >= 0")
    else:
        if strict and value <= 0:
            raise ValidationError(f"{name}: Must be > 0, got {value}")
        elif not strict and value < 0:
            raise ValidationError(f"{name}: Must be >= 0, got {value}")

    return value


def validate_range(
    value: Union[int, float, np.ndarray],
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    name: str = "value",
    inclusive: bool = True,
) -> Union[int, float, np.ndarray]:
    """
    Validate that value is within specified range.

    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name for error messages
        inclusive: Include boundary values

    Returns:
        Validated value

    Raises:
        ValidationError: If validation fails

    Example:
        >>> ph = validate_range(6.5, min_val=0, max_val=14, name="pH")
    """
    arr = np.atleast_1d(value)

    if min_val is not None:
        if inclusive:
            if np.any(arr < min_val):
                raise ValidationError(
                    f"{name}: Values must be >= {min_val}"
                )
        else:
            if np.any(arr <= min_val):
                raise ValidationError(
                    f"{name}: Values must be > {min_val}"
                )

    if max_val is not None:
        if inclusive:
            if np.any(arr > max_val):
                raise ValidationError(
                    f"{name}: Values must be <= {max_val}"
                )
        else:
            if np.any(arr >= max_val):
                raise ValidationError(
                    f"{name}: Values must be < {max_val}"
                )

    return value


def validate_coordinates(
    lat: Union[float, np.ndarray],
    lon: Union[float, np.ndarray],
    name: str = "coordinates",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate geographic coordinates.

    Args:
        lat: Latitude value(s) in degrees
        lon: Longitude value(s) in degrees
        name: Name for error messages

    Returns:
        Tuple of validated (lat, lon) arrays

    Raises:
        ValidationError: If coordinates are invalid

    Example:
        >>> lat, lon = validate_coordinates(45.5, -122.6, "farm_location")
    """
    lat = np.atleast_1d(lat)
    lon = np.atleast_1d(lon)

    if lat.shape != lon.shape:
        raise ValidationError(
            f"{name}: Latitude and longitude must have same shape"
        )

    if np.any(lat < -90) or np.any(lat > 90):
        raise ValidationError(
            f"{name}: Latitude must be between -90 and 90 degrees"
        )

    if np.any(lon < -180) or np.any(lon > 180):
        raise ValidationError(
            f"{name}: Longitude must be between -180 and 180 degrees"
        )

    return lat, lon


def validate_date_range(
    start_date: Union[str, datetime, date],
    end_date: Union[str, datetime, date],
    name: str = "date_range",
    date_format: str = "%Y-%m-%d",
) -> Tuple[datetime, datetime]:
    """
    Validate date range.

    Args:
        start_date: Start date
        end_date: End date
        name: Name for error messages
        date_format: Format for parsing string dates

    Returns:
        Tuple of (start, end) datetime objects

    Raises:
        ValidationError: If dates are invalid

    Example:
        >>> start, end = validate_date_range("2024-01-01", "2024-12-31")
    """
    # Convert to datetime
    if isinstance(start_date, str):
        try:
            start_date = datetime.strptime(start_date, date_format)
        except ValueError as e:
            raise ValidationError(f"{name}: Invalid start date format - {e}")

    if isinstance(end_date, str):
        try:
            end_date = datetime.strptime(end_date, date_format)
        except ValueError as e:
            raise ValidationError(f"{name}: Invalid end date format - {e}")

    if isinstance(start_date, date) and not isinstance(start_date, datetime):
        start_date = datetime.combine(start_date, datetime.min.time())

    if isinstance(end_date, date) and not isinstance(end_date, datetime):
        end_date = datetime.combine(end_date, datetime.max.time())

    # Check order
    if start_date > end_date:
        raise ValidationError(
            f"{name}: Start date must be before end date"
        )

    return start_date, end_date


class DataValidator:
    """
    Comprehensive data validator for agricultural datasets.

    Validates multi-column datasets with configurable rules
    for each column, detecting issues and providing reports.

    Example:
        >>> validator = DataValidator()
        >>> validator.add_rule("temperature", validate_range, min_val=-50, max_val=60)
        >>> validator.add_rule("humidity", validate_range, min_val=0, max_val=100)
        >>> result = validator.validate(weather_data)
    """

    def __init__(self, strict: bool = False):
        """
        Initialize data validator.

        Args:
            strict: Raise exception on first error if True
        """
        self.strict = strict
        self._rules: dict = {}
        self._custom_validators: List[Callable] = []

    def add_rule(
        self,
        column: str,
        validator: Callable,
        **kwargs,
    ) -> "DataValidator":
        """
        Add validation rule for a column.

        Args:
            column: Column name
            validator: Validation function
            **kwargs: Arguments for validator

        Returns:
            Self for method chaining
        """
        self._rules[column] = (validator, kwargs)
        return self

    def add_custom_validator(
        self,
        validator: Callable[[Any], ValidationResult],
    ) -> "DataValidator":
        """
        Add custom validation function.

        Args:
            validator: Function that returns ValidationResult

        Returns:
            Self for method chaining
        """
        self._custom_validators.append(validator)
        return self

    def validate(
        self,
        data: Any,
        columns: Optional[List[str]] = None,
    ) -> ValidationResult:
        """
        Validate data against all rules.

        Args:
            data: Data to validate (dict, DataFrame-like, or array)
            columns: Specific columns to validate (all if None)

        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []
        stats = {}

        # Convert data to dict-like access
        if hasattr(data, "columns"):  # DataFrame-like
            data_dict = {col: data[col].values for col in data.columns}
        elif isinstance(data, dict):
            data_dict = data
        elif isinstance(data, np.ndarray):
            if data.ndim == 2:
                data_dict = {f"col_{i}": data[:, i] for i in range(data.shape[1])}
            else:
                data_dict = {"data": data}
        else:
            data_dict = {"data": np.asarray(data)}

        # Apply column rules
        columns_to_validate = columns or list(data_dict.keys())

        for col in columns_to_validate:
            if col not in data_dict:
                warnings.append(f"Column '{col}' not found in data")
                continue

            col_data = data_dict[col]

            # Collect statistics
            if np.issubdtype(np.asarray(col_data).dtype, np.number):
                col_array = np.asarray(col_data)
                stats[col] = {
                    "count": len(col_array),
                    "missing": int(np.sum(np.isnan(col_array))) if np.issubdtype(col_array.dtype, np.floating) else 0,
                    "min": float(np.nanmin(col_array)),
                    "max": float(np.nanmax(col_array)),
                    "mean": float(np.nanmean(col_array)),
                }

            # Apply rule if exists
            if col in self._rules:
                validator, kwargs = self._rules[col]
                try:
                    validator(col_data, name=col, **kwargs)
                except ValidationError as e:
                    if self.strict:
                        raise
                    errors.append(str(e))

        # Apply custom validators
        for validator in self._custom_validators:
            try:
                result = validator(data)
                if not result.is_valid:
                    errors.extend(result.errors)
                warnings.extend(result.warnings)
            except Exception as e:
                errors.append(f"Custom validator failed: {e}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )

    def validate_schema(
        self,
        data: Any,
        required_columns: List[str],
        optional_columns: Optional[List[str]] = None,
    ) -> ValidationResult:
        """
        Validate that data has required columns/keys.

        Args:
            data: Data to validate
            required_columns: Required column names
            optional_columns: Optional column names

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        # Get column names
        if hasattr(data, "columns"):
            columns = list(data.columns)
        elif isinstance(data, dict):
            columns = list(data.keys())
        else:
            return ValidationResult(
                is_valid=False,
                errors=["Data must be dict or DataFrame-like"],
                warnings=[],
            )

        # Check required columns
        for col in required_columns:
            if col not in columns:
                errors.append(f"Required column '{col}' is missing")

        # Check for unknown columns
        all_known = set(required_columns + (optional_columns or []))
        unknown = set(columns) - all_known
        if unknown:
            warnings.append(f"Unknown columns: {unknown}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
