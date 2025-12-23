"""
Spatial Analysis Module for SmartAgri

Provides GIS and spatial analysis capabilities:
- Coordinate transformations
- Spatial interpolation
- Zone management
- Field boundary analysis

Example:
    >>> from smartagri.spatial import SpatialAnalyzer
    >>> analyzer = SpatialAnalyzer(crs='EPSG:4326')
    >>> zones = analyzer.create_management_zones(field_data)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

__all__ = [
    "SpatialAnalyzer",
    "CoordinateTransformer",
    "ZoneManager",
    "FieldBoundary",
]


@dataclass
class FieldBoundary:
    """Field boundary definition."""
    vertices: np.ndarray  # (n, 2) array of coordinates
    crs: str = "EPSG:4326"
    area_ha: float = 0.0


class CoordinateTransformer:
    """
    Coordinate reference system transformations.

    Transforms coordinates between different CRS.
    """

    def __init__(self, source_crs: str, target_crs: str):
        """Initialize transformer."""
        self.source_crs = source_crs
        self.target_crs = target_crs

    def transform(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform coordinates."""
        # Simplified passthrough (in production, use pyproj)
        return x.copy(), y.copy()


class SpatialAnalyzer:
    """
    Spatial data analysis for precision agriculture.

    Example:
        >>> analyzer = SpatialAnalyzer()
        >>> interpolated = analyzer.interpolate_idw(points, values, grid)
    """

    def __init__(self, crs: str = "EPSG:4326"):
        """Initialize spatial analyzer."""
        self.crs = crs

    def calculate_area(self, boundary: FieldBoundary) -> float:
        """Calculate field area in hectares."""
        vertices = boundary.vertices
        n = len(vertices)
        if n < 3:
            return 0.0

        # Shoelace formula
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i, 0] * vertices[j, 1]
            area -= vertices[j, 0] * vertices[i, 1]

        area = abs(area) / 2.0

        # Convert to hectares (assuming meters)
        return area / 10000

    def interpolate_idw(
        self,
        points: np.ndarray,
        values: np.ndarray,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        power: float = 2.0,
    ) -> np.ndarray:
        """
        Inverse Distance Weighting interpolation.

        Args:
            points: Sample point coordinates (n, 2)
            values: Sample values (n,)
            grid_x: X coordinates of interpolation grid
            grid_y: Y coordinates of interpolation grid
            power: Distance power parameter

        Returns:
            Interpolated values on grid
        """
        result = np.zeros((len(grid_y), len(grid_x)))

        for i, y in enumerate(grid_y):
            for j, x in enumerate(grid_x):
                distances = np.sqrt((points[:, 0] - x)**2 + (points[:, 1] - y)**2)

                # Handle exact matches
                zero_dist = distances < 1e-10
                if np.any(zero_dist):
                    result[i, j] = values[zero_dist][0]
                else:
                    weights = 1 / (distances ** power)
                    result[i, j] = np.sum(weights * values) / np.sum(weights)

        return result

    def create_management_zones(
        self,
        field_data: Dict[str, np.ndarray],
        n_zones: int = 3,
    ) -> np.ndarray:
        """Create management zones using clustering."""
        # Simple quantile-based zonation
        if 'yield' in field_data:
            values = field_data['yield']
        else:
            values = list(field_data.values())[0]

        percentiles = np.linspace(0, 100, n_zones + 1)
        thresholds = np.percentile(values, percentiles)

        zones = np.zeros_like(values, dtype=int)
        for i in range(n_zones):
            mask = (values >= thresholds[i]) & (values <= thresholds[i + 1])
            zones[mask] = i + 1

        return zones


class ZoneManager:
    """Management zone operations."""

    def __init__(self):
        """Initialize zone manager."""
        self.zones: Dict[int, Dict] = {}

    def create_zone(
        self,
        zone_id: int,
        boundary: np.ndarray,
        properties: Dict[str, Any] = None,
    ) -> None:
        """Create new management zone."""
        self.zones[zone_id] = {
            'boundary': boundary,
            'properties': properties or {},
        }

    def get_zone_statistics(self, zone_id: int) -> Dict[str, Any]:
        """Get statistics for zone."""
        return self.zones.get(zone_id, {}).get('properties', {})
