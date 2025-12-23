# -*- coding: utf-8 -*-
"""
Remote Sensing Analysis Example
================================

This example demonstrates remote sensing analysis using the hetao_ag library,
including spectral index calculation, vegetation health assessment, and
crop phenology classification.

Learning Objectives:
- Calculate vegetation indices (NDVI, SAVI, EVI, LSWI, NDWI)
- Classify vegetation health status
- Track crop phenology from time series
- Perform basic image processing operations

Author: Hetao College
"""

import numpy as np
from typing import Tuple

# Import hetao_ag modules
from hetao_ag.space import (
    compute_ndvi,
    compute_savi,
    compute_evi,
    compute_lswi,
    compute_ndwi,
    classify_vegetation_health,
    RasterImage,
    GeoMetadata,
    CloudMask,
    PhenologyClassifier,
    PhenologyFeatures,
    temporal_smoothing
)


def demonstrate_spectral_indices():
    """
    Demonstrate calculation of various spectral vegetation indices.
    """
    print("=" * 70)
    print("SPECTRAL VEGETATION INDICES")
    print("=" * 70)

    # Simulate band data (typical Sentinel-2 values scaled 0-10000)
    # Create a 5x5 pixel grid with varying vegetation cover
    np.random.seed(42)

    # Create gradient from bare soil (low vegetation) to dense vegetation
    vegetation_gradient = np.linspace(0.1, 0.9, 25).reshape(5, 5)

    # Typical reflectance values:
    # Bare soil: high red, low NIR
    # Dense vegetation: low red, high NIR

    # Blue band (B2)
    blue = (800 + 200 * (1 - vegetation_gradient) +
            np.random.randn(5, 5) * 50).astype(np.uint16)

    # Green band (B3)
    green = (700 + 300 * (1 - vegetation_gradient) +
             np.random.randn(5, 5) * 50).astype(np.uint16)

    # Red band (B4)
    red = (1200 - 600 * vegetation_gradient +
           np.random.randn(5, 5) * 50).astype(np.uint16)
    red = np.clip(red, 200, 2000).astype(np.uint16)

    # NIR band (B8)
    nir = (1500 + 2500 * vegetation_gradient +
           np.random.randn(5, 5) * 100).astype(np.uint16)

    # SWIR band (B11)
    swir = (2000 + 500 * (1 - vegetation_gradient) +
            np.random.randn(5, 5) * 100).astype(np.uint16)

    print("\nSimulated 5x5 pixel scene with vegetation gradient...")
    print("(Low vegetation top-left, high vegetation bottom-right)\n")

    # =========================================================================
    # Calculate NDVI
    # =========================================================================
    print("1. NDVI (Normalized Difference Vegetation Index)")
    print("   Formula: (NIR - Red) / (NIR + Red)")
    print("   Range: -1 to 1 (higher = more vegetation)")
    print("-" * 50)

    ndvi = compute_ndvi(red, nir)
    print(f"   NDVI values:\n{ndvi.round(3)}")
    print(f"   Min: {ndvi.min():.3f}, Max: {ndvi.max():.3f}, Mean: {ndvi.mean():.3f}")

    # =========================================================================
    # Calculate SAVI
    # =========================================================================
    print("\n2. SAVI (Soil-Adjusted Vegetation Index)")
    print("   Formula: ((NIR - Red) / (NIR + Red + L)) * (1 + L)")
    print("   L=0.5 for intermediate vegetation cover")
    print("-" * 50)

    savi = compute_savi(red, nir, L=0.5)
    print(f"   SAVI values:\n{savi.round(3)}")
    print(f"   Min: {savi.min():.3f}, Max: {savi.max():.3f}, Mean: {savi.mean():.3f}")

    # =========================================================================
    # Calculate EVI
    # =========================================================================
    print("\n3. EVI (Enhanced Vegetation Index)")
    print("   Formula: G * (NIR - Red) / (NIR + C1*Red - C2*Blue + L)")
    print("   Reduces atmospheric and soil background effects")
    print("-" * 50)

    evi = compute_evi(blue, red, nir)
    print(f"   EVI values:\n{evi.round(3)}")
    print(f"   Min: {evi.min():.3f}, Max: {evi.max():.3f}, Mean: {evi.mean():.3f}")

    # =========================================================================
    # Calculate LSWI
    # =========================================================================
    print("\n4. LSWI (Land Surface Water Index)")
    print("   Formula: (NIR - SWIR) / (NIR + SWIR)")
    print("   Sensitive to vegetation water content")
    print("-" * 50)

    lswi = compute_lswi(nir, swir)
    print(f"   LSWI values:\n{lswi.round(3)}")
    print(f"   Min: {lswi.min():.3f}, Max: {lswi.max():.3f}, Mean: {lswi.mean():.3f}")

    # =========================================================================
    # Calculate NDWI
    # =========================================================================
    print("\n5. NDWI (Normalized Difference Water Index)")
    print("   Formula: (Green - NIR) / (Green + NIR)")
    print("   Positive values indicate water bodies")
    print("-" * 50)

    ndwi = compute_ndwi(green, nir)
    print(f"   NDWI values:\n{ndwi.round(3)}")
    print(f"   Min: {ndwi.min():.3f}, Max: {ndwi.max():.3f}, Mean: {ndwi.mean():.3f}")

    # Return indices for further analysis
    return ndvi, savi, evi, lswi, ndwi


def demonstrate_vegetation_classification():
    """
    Demonstrate vegetation health classification based on NDVI.
    """
    print("\n" + "=" * 70)
    print("VEGETATION HEALTH CLASSIFICATION")
    print("=" * 70)

    print("\nNDVI-based vegetation health classes:")
    print("-" * 50)

    ndvi_samples = [-0.2, 0.0, 0.1, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85]

    for ndvi in ndvi_samples:
        health_class = classify_vegetation_health(ndvi)
        # Create visual bar
        bar_length = int((ndvi + 0.2) * 30)  # Normalize for display
        bar = "|" + "#" * max(0, bar_length) + " " * (35 - bar_length) + "|"
        print(f"   NDVI = {ndvi:>5.2f}  {bar}  {health_class}")

    print("\nClassification thresholds:")
    print("   < 0.0  : Water/Bare soil")
    print("   0.0-0.2: Sparse or no vegetation")
    print("   0.2-0.4: Light vegetation")
    print("   0.4-0.6: Moderate vegetation")
    print("   0.6-0.8: Dense vegetation")
    print("   > 0.8  : Very dense vegetation")


def demonstrate_image_processing():
    """
    Demonstrate RasterImage class operations.
    """
    print("\n" + "=" * 70)
    print("RASTER IMAGE PROCESSING")
    print("=" * 70)

    # Create a synthetic multi-band image
    np.random.seed(42)
    height, width = 100, 100
    n_bands = 4

    # Simulate bands with spatial patterns
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))

    # Blue band - higher in water areas (bottom-right)
    blue = (1000 + 500 * np.sqrt((x-0.8)**2 + (y-0.8)**2) +
            np.random.randn(height, width) * 100).astype(np.uint16)

    # Green band
    green = (800 + 300 * np.sin(x * 3.14) +
             np.random.randn(height, width) * 80).astype(np.uint16)

    # Red band - lower in vegetated areas (top-left)
    red = (600 + 400 * np.sqrt(x**2 + y**2) +
           np.random.randn(height, width) * 80).astype(np.uint16)

    # NIR band - higher in vegetated areas
    nir = (3000 - 1500 * np.sqrt(x**2 + y**2) +
           np.random.randn(height, width) * 150).astype(np.uint16)

    # Stack bands
    data = np.stack([blue, green, red, nir])

    # Create RasterImage
    img = RasterImage(
        data=data,
        band_names={"blue": 0, "green": 1, "red": 2, "nir": 3},
        metadata=GeoMetadata(
            crs="EPSG:32650",
            resolution=10.0
        )
    )

    print(f"\n1. Image Properties:")
    print(f"   Shape: {img.shape} (bands, height, width)")
    print(f"   Number of bands: {img.n_bands}")
    print(f"   CRS: {img.metadata.crs}")
    print(f"   Resolution: {img.metadata.resolution} m")

    # Get bands by name
    print(f"\n2. Band Access:")
    red_band = img.get_band("red")
    nir_band = img.get_band("nir")
    print(f"   Red band shape: {red_band.shape}")
    print(f"   Red band range: {red_band.min()} - {red_band.max()}")
    print(f"   NIR band range: {nir_band.min()} - {nir_band.max()}")

    # Subset operation
    print(f"\n3. Subset Operation:")
    subset = img.subset(slice(25, 75), slice(25, 75))
    print(f"   Original shape: {img.shape}")
    print(f"   Subset shape: {subset.shape}")

    # Cloud masking
    print(f"\n4. Cloud Masking:")
    cloud_mask_gen = CloudMask(threshold=0.3)

    # Simulate QA band (high values = clouds)
    qa_band = np.zeros((height, width), dtype=np.uint8)
    qa_band[60:80, 60:80] = 255  # Cloud patch

    valid_mask = cloud_mask_gen.from_qa_band(qa_band)
    cloud_pixels = np.sum(~valid_mask)
    total_pixels = valid_mask.size
    print(f"   Cloudy pixels: {cloud_pixels} ({cloud_pixels/total_pixels*100:.1f}%)")

    # Calculate NDVI for entire image
    print(f"\n5. Full Image NDVI:")
    ndvi = compute_ndvi(red_band, nir_band)
    print(f"   NDVI shape: {ndvi.shape}")
    print(f"   NDVI range: {ndvi.min():.3f} to {ndvi.max():.3f}")
    print(f"   Mean NDVI: {ndvi.mean():.3f}")

    # Apply mask to NDVI
    masked_ndvi = np.where(valid_mask, ndvi, np.nan)
    print(f"   Mean NDVI (cloud-free): {np.nanmean(masked_ndvi):.3f}")

    return img, ndvi


def demonstrate_phenology_classification():
    """
    Demonstrate phenology classification from NDVI time series.
    """
    print("\n" + "=" * 70)
    print("PHENOLOGY CLASSIFICATION FROM NDVI TIME SERIES")
    print("=" * 70)

    # Create synthetic NDVI time series (12 months, 50x50 pixels)
    np.random.seed(42)
    n_times = 12
    height, width = 50, 50

    # Time axis (months)
    t = np.linspace(0, 2 * np.pi, n_times)

    print("\nCreating synthetic crop phenology patterns...")

    # Create different crop types with different phenology
    # Winter wheat: peaks in May-June (month 5-6)
    # Summer maize: peaks in August-September (month 8-9)
    # Mixed/perennial: relatively stable

    ndvi_series = np.zeros((n_times, height, width))

    # Create crop type zones
    crop_zones = np.zeros((height, width), dtype=int)
    crop_zones[:20, :] = 0      # Winter wheat (top)
    crop_zones[20:35, :] = 1    # Summer maize (middle)
    crop_zones[35:, :] = 2      # Mixed/bare (bottom)

    for i in range(height):
        for j in range(width):
            zone = crop_zones[i, j]

            if zone == 0:  # Winter wheat - early peak
                base_curve = 0.3 + 0.4 * np.sin(t - 0.5)  # Peak at ~month 4
            elif zone == 1:  # Summer maize - late peak
                base_curve = 0.25 + 0.45 * np.sin(t + 1.5)  # Peak at ~month 8
            else:  # Mixed/bare - low and stable
                base_curve = 0.2 + 0.1 * np.sin(t)

            # Add noise and spatial variation
            noise = np.random.randn(n_times) * 0.05
            spatial_var = (np.random.rand() - 0.5) * 0.1

            ndvi_series[:, i, j] = np.clip(base_curve + noise + spatial_var, 0, 1)

    print(f"   Created NDVI time series: {ndvi_series.shape}")
    print(f"   (12 months, 50x50 pixels)")

    # Apply temporal smoothing
    print("\n1. Temporal Smoothing:")
    smoothed = temporal_smoothing(ndvi_series, window=3)
    print(f"   Applied 3-month moving window smoothing")
    print(f"   Original variance: {ndvi_series.var():.4f}")
    print(f"   Smoothed variance: {smoothed.var():.4f}")

    # Create phenology classifier
    print("\n2. Phenology Classification:")
    classifier = PhenologyClassifier(smoothed)

    # Classify crops
    crop_map = classifier.classify_crops(n_classes=3)

    # Count pixels per class
    print("   Classification results:")
    for class_id in range(4):  # 0 = non-crop, 1-3 = crop types
        count = np.sum(crop_map == class_id)
        percentage = count / crop_map.size * 100
        if class_id == 0:
            label = "Non-crop/low vegetation"
        elif class_id == 1:
            label = "Early-season crop (winter wheat)"
        elif class_id == 2:
            label = "Late-season crop (summer maize)"
        else:
            label = "Mid-season crop"
        print(f"   Class {class_id}: {count:>5} pixels ({percentage:>5.1f}%) - {label}")

    # Extract features for sample pixels
    print("\n3. Phenology Features for Sample Pixels:")
    sample_points = [(10, 25), (27, 25), (42, 25)]  # One from each zone
    zone_names = ["Winter wheat zone", "Summer maize zone", "Low vegetation zone"]

    for (row, col), zone_name in zip(sample_points, zone_names):
        features = classifier.extract_features(row, col)
        print(f"\n   {zone_name} (row={row}, col={col}):")
        print(f"     Peak NDVI: {features.peak_value:.3f}")
        print(f"     Peak time: Month {features.peak_time + 1}")
        print(f"     Season start: Month {features.start_of_season + 1}")
        print(f"     Season end: Month {features.end_of_season + 1}")
        print(f"     Amplitude: {features.amplitude:.3f}")

    # Get phenology maps
    print("\n4. Generating Phenology Maps:")
    pheno_maps = classifier.get_phenology_map()
    for map_name, map_data in pheno_maps.items():
        print(f"   {map_name}: shape={map_data.shape}, "
              f"range=[{map_data.min():.2f}, {map_data.max():.2f}]")

    return ndvi_series, classifier, crop_map


def practical_application_example():
    """
    Demonstrate a practical remote sensing workflow.
    """
    print("\n" + "=" * 70)
    print("PRACTICAL APPLICATION: CROP MONITORING WORKFLOW")
    print("=" * 70)

    print("""
    This workflow demonstrates how to use hetao_ag for crop monitoring:

    1. Load satellite image (simulated here)
    2. Apply cloud masking
    3. Calculate vegetation indices
    4. Classify vegetation health
    5. Identify stressed areas
    6. Generate monitoring report
    """)

    # Step 1: Create simulated Sentinel-2 scene
    print("Step 1: Loading satellite image...")
    np.random.seed(42)

    # 200x200 pixel scene, ~2km x 2km at 10m resolution
    height, width = 200, 200

    # Create field patterns (4 fields in quadrants)
    fields = np.zeros((height, width))
    fields[:100, :100] = 0.7   # Healthy field
    fields[:100, 100:] = 0.5   # Moderate stress
    fields[100:, :100] = 0.3   # Severe stress
    fields[100:, 100:] = 0.6   # Slight stress

    # Add noise and field boundaries
    fields += np.random.randn(height, width) * 0.05
    fields = np.clip(fields, 0, 1)

    # Generate bands based on field condition
    red = (1000 - 600 * fields + np.random.randn(height, width) * 50).astype(np.uint16)
    nir = (1500 + 2500 * fields + np.random.randn(height, width) * 100).astype(np.uint16)

    print(f"   Image size: {height}x{width} pixels (10m resolution)")

    # Step 2: Cloud masking
    print("\nStep 2: Applying cloud mask...")
    # Add cloud patch
    cloud_mask = np.ones((height, width), dtype=bool)
    cloud_mask[150:180, 50:100] = False  # Cloud patch

    cloudy_pct = 100 * np.sum(~cloud_mask) / cloud_mask.size
    print(f"   Cloud cover: {cloudy_pct:.1f}%")

    # Step 3: Calculate NDVI
    print("\nStep 3: Calculating NDVI...")
    ndvi = compute_ndvi(red, nir)
    ndvi_masked = np.where(cloud_mask, ndvi, np.nan)
    print(f"   Mean NDVI (cloud-free): {np.nanmean(ndvi_masked):.3f}")

    # Step 4: Classify vegetation health
    print("\nStep 4: Classifying vegetation health by field...")

    field_stats = [
        ("Northwest (healthy)", 0, 100, 0, 100),
        ("Northeast (moderate)", 0, 100, 100, 200),
        ("Southwest (stressed)", 100, 200, 0, 100),
        ("Southeast (slight)", 100, 200, 100, 200),
    ]

    print(f"\n   {'Field':<25} {'Mean NDVI':>10} {'Health Status'}")
    print("   " + "-" * 55)

    for name, r1, r2, c1, c2 in field_stats:
        field_ndvi = ndvi_masked[r1:r2, c1:c2]
        mean_ndvi = np.nanmean(field_ndvi)
        health = classify_vegetation_health(mean_ndvi)
        print(f"   {name:<25} {mean_ndvi:>10.3f} {health}")

    # Step 5: Identify stressed areas
    print("\nStep 5: Identifying stressed areas...")
    stress_threshold = 0.4
    stressed = (ndvi_masked < stress_threshold) & cloud_mask
    stressed_area = np.sum(stressed) * 100  # m2 (10m pixels)
    total_area = np.sum(cloud_mask) * 100

    print(f"   Stress threshold: NDVI < {stress_threshold}")
    print(f"   Stressed area: {stressed_area/10000:.2f} ha")
    print(f"   Total monitored: {total_area/10000:.2f} ha")
    print(f"   Stressed percentage: {100*stressed_area/total_area:.1f}%")

    # Step 6: Generate report
    print("\n" + "=" * 70)
    print("MONITORING REPORT SUMMARY")
    print("=" * 70)
    print(f"""
    Date: [Current Date]
    Location: Hetao Irrigation District
    Image Source: Sentinel-2 (simulated)

    OVERVIEW:
    - Scene size: {width*10/1000:.1f} km x {height*10/1000:.1f} km
    - Cloud cover: {cloudy_pct:.1f}%
    - Mean NDVI: {np.nanmean(ndvi_masked):.3f}

    FIELD STATUS:
    - Healthy vegetation (NDVI > 0.6): 1 field (NW quadrant)
    - Moderate condition (NDVI 0.4-0.6): 2 fields (NE, SE)
    - Stressed vegetation (NDVI < 0.4): 1 field (SW quadrant)

    RECOMMENDATIONS:
    - SW quadrant requires immediate inspection
    - Consider irrigation scheduling adjustment for stressed areas
    - Schedule follow-up monitoring in 7-10 days

    END OF REPORT
    """)


def main():
    """Run all demonstrations."""

    print("\n" + "=" * 70)
    print("REMOTE SENSING ANALYSIS - COMPREHENSIVE EXAMPLE")
    print("hetao_ag Library Demonstration")
    print("=" * 70)

    # Run demonstrations
    indices = demonstrate_spectral_indices()
    demonstrate_vegetation_classification()
    img, ndvi = demonstrate_image_processing()
    ndvi_series, classifier, crop_map = demonstrate_phenology_classification()
    practical_application_example()

    print("\n" + "=" * 70)
    print("Example Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. NDVI is fundamental but SAVI/EVI may be better for sparse vegetation")
    print("  2. LSWI is useful for monitoring crop water stress")
    print("  3. Phenology classification helps identify different crop types")
    print("  4. Time series smoothing reduces noise and improves classification")
    print("  5. Cloud masking is essential for accurate analysis")


if __name__ == "__main__":
    main()
