"""
Remote Sensing Imagery Module

Provides satellite and drone imagery processing:
- Vegetation index calculation
- Image classification
- Change detection
- Crop health assessment

Example:
    >>> from smartagri.imagery import VegetationIndices
    >>> vi = VegetationIndices()
    >>> ndvi = vi.calculate_ndvi(nir_band, red_band)
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

__all__ = [
    "VegetationIndices",
    "ImageClassifier",
    "ChangeDetector",
    "ImagePreprocessor",
]


@dataclass
class BandData:
    """Spectral band data container."""
    data: np.ndarray
    wavelength_nm: float
    band_name: str


class VegetationIndices:
    """
    Vegetation index calculations.

    Computes various vegetation indices from
    multispectral imagery.

    Example:
        >>> vi = VegetationIndices()
        >>> ndvi = vi.calculate_ndvi(nir, red)
        >>> evi = vi.calculate_evi(nir, red, blue)
    """

    def calculate_ndvi(
        self,
        nir: np.ndarray,
        red: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate Normalized Difference Vegetation Index.

        NDVI = (NIR - Red) / (NIR + Red)

        Args:
            nir: Near-infrared band
            red: Red band

        Returns:
            NDVI array (-1 to 1)
        """
        nir = nir.astype(float)
        red = red.astype(float)

        denominator = nir + red
        ndvi = np.where(
            denominator != 0,
            (nir - red) / denominator,
            0
        )

        return np.clip(ndvi, -1, 1)

    def calculate_evi(
        self,
        nir: np.ndarray,
        red: np.ndarray,
        blue: np.ndarray,
        g: float = 2.5,
        c1: float = 6.0,
        c2: float = 7.5,
        l: float = 1.0,
    ) -> np.ndarray:
        """
        Calculate Enhanced Vegetation Index.

        EVI = G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L)

        Args:
            nir: Near-infrared band
            red: Red band
            blue: Blue band
            g, c1, c2, l: EVI coefficients

        Returns:
            EVI array
        """
        nir = nir.astype(float)
        red = red.astype(float)
        blue = blue.astype(float)

        denominator = nir + c1 * red - c2 * blue + l
        evi = np.where(
            denominator != 0,
            g * (nir - red) / denominator,
            0
        )

        return np.clip(evi, -1, 1)

    def calculate_savi(
        self,
        nir: np.ndarray,
        red: np.ndarray,
        l: float = 0.5,
    ) -> np.ndarray:
        """
        Calculate Soil Adjusted Vegetation Index.

        SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)

        Args:
            nir: Near-infrared band
            red: Red band
            l: Soil brightness correction factor

        Returns:
            SAVI array
        """
        nir = nir.astype(float)
        red = red.astype(float)

        denominator = nir + red + l
        savi = np.where(
            denominator != 0,
            ((nir - red) / denominator) * (1 + l),
            0
        )

        return savi

    def calculate_ndwi(
        self,
        green: np.ndarray,
        nir: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate Normalized Difference Water Index.

        NDWI = (Green - NIR) / (Green + NIR)

        Args:
            green: Green band
            nir: Near-infrared band

        Returns:
            NDWI array
        """
        green = green.astype(float)
        nir = nir.astype(float)

        denominator = green + nir
        ndwi = np.where(
            denominator != 0,
            (green - nir) / denominator,
            0
        )

        return np.clip(ndwi, -1, 1)


class ImageClassifier:
    """
    Supervised image classification.

    Classifies imagery into land cover or crop types.
    """

    def __init__(self, n_classes: int = 5):
        """Initialize classifier."""
        self.n_classes = n_classes
        self._is_trained = False

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Train classifier on labeled data."""
        self._is_trained = True

    def classify(self, image: np.ndarray) -> np.ndarray:
        """Classify image pixels."""
        if not self._is_trained:
            raise ValueError("Classifier not trained")

        # Random classification for demo
        return np.random.randint(0, self.n_classes, image.shape[:2])


class ChangeDetector:
    """
    Multi-temporal change detection.

    Detects changes between image dates.
    """

    def detect_changes(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        threshold: float = 0.2,
    ) -> np.ndarray:
        """
        Detect changes between two images.

        Args:
            image1: First image
            image2: Second image
            threshold: Change threshold

        Returns:
            Binary change mask
        """
        diff = np.abs(image2.astype(float) - image1.astype(float))

        if len(diff.shape) > 2:
            diff = np.mean(diff, axis=2)

        return diff > threshold


class ImagePreprocessor:
    """Image preprocessing utilities."""

    def radiometric_calibration(
        self,
        raw_dn: np.ndarray,
        gain: float = 1.0,
        offset: float = 0.0,
    ) -> np.ndarray:
        """Convert DN to radiance."""
        return raw_dn * gain + offset

    def atmospheric_correction(
        self,
        radiance: np.ndarray,
        dark_object: float = None,
    ) -> np.ndarray:
        """Simple dark object subtraction correction."""
        if dark_object is None:
            dark_object = np.percentile(radiance, 1)
        return np.maximum(radiance - dark_object, 0)
