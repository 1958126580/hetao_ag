"""
Soil Mapping Module

Spatial soil analysis and mapping tools:
- Kriging interpolation
- Management zone delineation
- Variability analysis
- Precision sampling design

Example:
    >>> mapper = SoilMapper()
    >>> grid_values = mapper.interpolate(sample_points, sample_values, grid)
    >>> zones = mapper.create_management_zones(grid_values, n_zones=3)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ManagementZone:
    """
    Agricultural management zone definition.

    Attributes:
        zone_id: Unique zone identifier
        mean_value: Mean value within zone
        std_value: Standard deviation within zone
        area: Zone area (ha)
        boundary: Zone boundary coordinates
        recommendations: Zone-specific recommendations
    """
    zone_id: int
    mean_value: float
    std_value: float
    area: float
    boundary: Optional[np.ndarray] = None
    recommendations: Dict[str, Any] = None


class SpatialInterpolation:
    """
    Spatial interpolation methods for soil properties.

    Implements kriging and inverse distance weighting for
    creating continuous soil property maps from point samples.
    """

    def __init__(self, method: str = "ordinary_kriging"):
        """
        Initialize spatial interpolation.

        Args:
            method: Interpolation method
        """
        self.method = method
        self._variogram_params = None

    def fit_variogram(
        self,
        coords: np.ndarray,
        values: np.ndarray,
        n_bins: int = 15,
    ) -> Dict[str, float]:
        """
        Fit variogram model to sample data.

        Args:
            coords: Sample coordinates (n, 2)
            values: Sample values (n,)
            n_bins: Number of distance bins

        Returns:
            Variogram parameters
        """
        n = len(values)

        # Calculate empirical variogram
        distances = []
        semivariances = []

        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
                sv = 0.5 * (values[i] - values[j]) ** 2
                distances.append(d)
                semivariances.append(sv)

        distances = np.array(distances)
        semivariances = np.array(semivariances)

        # Bin by distance
        max_dist = np.percentile(distances, 60)
        bins = np.linspace(0, max_dist, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        binned_sv = np.zeros(n_bins)
        counts = np.zeros(n_bins)

        for d, sv in zip(distances, semivariances):
            if d <= max_dist:
                idx = min(int(d / max_dist * n_bins), n_bins - 1)
                binned_sv[idx] += sv
                counts[idx] += 1

        valid = counts > 0
        binned_sv[valid] /= counts[valid]

        # Fit spherical model
        nugget = binned_sv[valid][0] if np.any(valid) else 0
        sill = np.max(binned_sv[valid]) if np.any(valid) else np.var(values)
        range_param = bin_centers[np.argmax(binned_sv[valid])] if np.any(valid) else max_dist / 2

        self._variogram_params = {
            "nugget": max(0, nugget),
            "sill": max(nugget, sill) - nugget,
            "range": range_param,
            "model": "spherical",
        }

        return self._variogram_params

    def interpolate(
        self,
        sample_coords: np.ndarray,
        sample_values: np.ndarray,
        target_coords: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate values at target locations.

        Args:
            sample_coords: Sample point coordinates
            sample_values: Values at sample points
            target_coords: Target locations for interpolation

        Returns:
            Tuple of (interpolated_values, estimation_variance)
        """
        if self._variogram_params is None:
            self.fit_variogram(sample_coords, sample_values)

        if self.method == "ordinary_kriging":
            return self._ordinary_kriging(
                sample_coords, sample_values, target_coords
            )
        else:
            return self._idw(sample_coords, sample_values, target_coords)

    def _ordinary_kriging(
        self,
        sample_coords: np.ndarray,
        sample_values: np.ndarray,
        target_coords: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ordinary kriging interpolation."""
        n = len(sample_values)
        m = len(target_coords)

        params = self._variogram_params
        nugget = params["nugget"]
        sill = params["sill"]
        a = params["range"]

        def covariance(h):
            """Spherical covariance function."""
            c0 = nugget + sill
            if h >= a:
                return 0
            return c0 - nugget - sill * (1.5 * h / a - 0.5 * (h / a) ** 3)

        # Build kriging matrix
        K = np.zeros((n + 1, n + 1))
        for i in range(n):
            for j in range(n):
                h = np.sqrt(np.sum((sample_coords[i] - sample_coords[j]) ** 2))
                K[i, j] = covariance(h)
        K[:n, n] = 1
        K[n, :n] = 1

        # Predictions
        predictions = np.zeros(m)
        variances = np.zeros(m)

        for t in range(m):
            k = np.zeros(n + 1)
            for i in range(n):
                h = np.sqrt(np.sum((sample_coords[i] - target_coords[t]) ** 2))
                k[i] = covariance(h)
            k[n] = 1

            try:
                weights = np.linalg.solve(K, k)
            except np.linalg.LinAlgError:
                weights = np.linalg.lstsq(K, k, rcond=None)[0]

            predictions[t] = np.dot(weights[:n], sample_values)
            variances[t] = max(0, (nugget + sill) - np.dot(weights, k))

        return predictions, variances

    def _idw(
        self,
        sample_coords: np.ndarray,
        sample_values: np.ndarray,
        target_coords: np.ndarray,
        power: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Inverse distance weighting interpolation."""
        m = len(target_coords)
        predictions = np.zeros(m)
        variances = np.zeros(m)

        for t in range(m):
            distances = np.sqrt(np.sum((sample_coords - target_coords[t]) ** 2, axis=1))

            # Handle zero distance
            if np.any(distances == 0):
                idx = np.argmin(distances)
                predictions[t] = sample_values[idx]
                variances[t] = 0
            else:
                weights = 1 / (distances ** power)
                weights /= weights.sum()
                predictions[t] = np.dot(weights, sample_values)
                variances[t] = np.dot(weights, (sample_values - predictions[t]) ** 2)

        return predictions, variances


class VariabilityAnalysis:
    """
    Analyze spatial variability of soil properties.
    """

    def coefficient_of_variation(self, values: np.ndarray) -> float:
        """Calculate coefficient of variation."""
        return np.std(values) / np.mean(values) * 100 if np.mean(values) != 0 else 0

    def spatial_dependency(
        self,
        variogram_params: Dict[str, float],
    ) -> str:
        """
        Classify spatial dependency based on variogram.

        Args:
            variogram_params: Variogram parameters

        Returns:
            Spatial dependency classification
        """
        nugget = variogram_params["nugget"]
        sill = variogram_params["sill"]

        nugget_ratio = nugget / (nugget + sill) if (nugget + sill) > 0 else 1

        if nugget_ratio < 0.25:
            return "strong"
        elif nugget_ratio < 0.75:
            return "moderate"
        else:
            return "weak"


class ManagementZones:
    """
    Create and analyze management zones for precision agriculture.

    Example:
        >>> zones = ManagementZones()
        >>> zone_map = zones.create_zones(grid_data, n_zones=3)
    """

    def create_zones(
        self,
        data: np.ndarray,
        n_zones: int = 3,
        method: str = "kmeans",
    ) -> np.ndarray:
        """
        Create management zones from spatial data.

        Args:
            data: 2D array of soil/yield data
            n_zones: Number of zones to create
            method: Clustering method

        Returns:
            Zone assignment array
        """
        # Flatten and remove NaN
        valid_mask = ~np.isnan(data)
        valid_values = data[valid_mask].reshape(-1, 1)

        # K-means clustering
        zone_labels = self._kmeans(valid_values, n_zones)

        # Create zone map
        zone_map = np.full(data.shape, np.nan)
        zone_map[valid_mask] = zone_labels

        return zone_map

    def _kmeans(
        self,
        data: np.ndarray,
        k: int,
        max_iter: int = 100,
    ) -> np.ndarray:
        """Simple k-means clustering."""
        n = len(data)

        # Initialize centroids
        idx = np.random.choice(n, k, replace=False)
        centroids = data[idx].flatten()

        for _ in range(max_iter):
            # Assign points to nearest centroid
            distances = np.abs(data.flatten()[:, np.newaxis] - centroids)
            labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.array([
                data[labels == i].mean() if np.sum(labels == i) > 0 else centroids[i]
                for i in range(k)
            ])

            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        return labels

    def analyze_zones(
        self,
        data: np.ndarray,
        zone_map: np.ndarray,
        pixel_size: float = 10.0,
    ) -> List[ManagementZone]:
        """
        Analyze management zones.

        Args:
            data: Original data array
            zone_map: Zone assignment array
            pixel_size: Pixel size in meters

        Returns:
            List of ManagementZone objects
        """
        zones = []
        unique_zones = np.unique(zone_map[~np.isnan(zone_map)])

        for zone_id in unique_zones:
            mask = zone_map == zone_id
            zone_values = data[mask]

            # Calculate area (convert m2 to ha)
            n_pixels = np.sum(mask)
            area = n_pixels * (pixel_size ** 2) / 10000

            zones.append(ManagementZone(
                zone_id=int(zone_id),
                mean_value=np.nanmean(zone_values),
                std_value=np.nanstd(zone_values),
                area=area,
            ))

        return zones


class SoilMapper:
    """
    High-level soil mapping interface.

    Combines interpolation, variability analysis, and zone creation
    for complete soil mapping workflow.
    """

    def __init__(self):
        """Initialize soil mapper."""
        self.interpolator = SpatialInterpolation()
        self.variability = VariabilityAnalysis()
        self.zone_creator = ManagementZones()

    def create_soil_map(
        self,
        sample_coords: np.ndarray,
        sample_values: np.ndarray,
        grid_resolution: float = 10.0,
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> Dict[str, Any]:
        """
        Create interpolated soil map from samples.

        Args:
            sample_coords: Sample point coordinates
            sample_values: Values at sample points
            grid_resolution: Grid cell size (m)
            bounds: Map bounds (xmin, ymin, xmax, ymax)

        Returns:
            Dict with interpolated map and metadata
        """
        # Determine bounds
        if bounds is None:
            xmin = sample_coords[:, 0].min() - grid_resolution
            xmax = sample_coords[:, 0].max() + grid_resolution
            ymin = sample_coords[:, 1].min() - grid_resolution
            ymax = sample_coords[:, 1].max() + grid_resolution
        else:
            xmin, ymin, xmax, ymax = bounds

        # Create grid
        x = np.arange(xmin, xmax, grid_resolution)
        y = np.arange(ymin, ymax, grid_resolution)
        xx, yy = np.meshgrid(x, y)
        grid_coords = np.column_stack([xx.ravel(), yy.ravel()])

        # Fit variogram and interpolate
        variogram = self.interpolator.fit_variogram(sample_coords, sample_values)
        predictions, variances = self.interpolator.interpolate(
            sample_coords, sample_values, grid_coords
        )

        # Reshape to grid
        pred_grid = predictions.reshape(xx.shape)
        var_grid = variances.reshape(xx.shape)

        # Variability analysis
        cv = self.variability.coefficient_of_variation(sample_values)
        spatial_dep = self.variability.spatial_dependency(variogram)

        return {
            "values": pred_grid,
            "variance": var_grid,
            "x": x,
            "y": y,
            "variogram": variogram,
            "cv": cv,
            "spatial_dependency": spatial_dep,
        }

    def interpolate(
        self,
        sample_coords: np.ndarray,
        sample_values: np.ndarray,
        target_coords: np.ndarray,
    ) -> np.ndarray:
        """Quick interpolation interface."""
        predictions, _ = self.interpolator.interpolate(
            sample_coords, sample_values, target_coords
        )
        return predictions

    def create_management_zones(
        self,
        grid_data: np.ndarray,
        n_zones: int = 3,
    ) -> np.ndarray:
        """Create management zones from grid data."""
        return self.zone_creator.create_zones(grid_data, n_zones)
