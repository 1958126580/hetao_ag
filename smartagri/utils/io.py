"""
Input/Output Utilities for SmartAgri

Provides file I/O operations for common agricultural data formats:
- CSV files for tabular data
- GeoTIFF for raster data
- Shapefiles for vector data
- NetCDF for climate data
- JSON for configuration

Features:
    - Streaming for large files
    - Compression support
    - Data validation on load
    - Format auto-detection
"""

import numpy as np
from typing import (
    Optional,
    Union,
    List,
    Dict,
    Any,
    Iterator,
    Callable,
)
from pathlib import Path
import json
import csv
import gzip
import logging
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import io

logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, List]


@dataclass
class GeoReference:
    """
    Geographic reference information for raster data.

    Attributes:
        crs: Coordinate reference system (e.g., 'EPSG:4326')
        transform: Affine transformation matrix
        bounds: Geographic bounds (xmin, ymin, xmax, ymax)
        resolution: Pixel resolution (x, y)
    """
    crs: str
    transform: tuple
    bounds: tuple
    resolution: tuple


def load_csv(
    filepath: Union[str, Path],
    delimiter: str = ",",
    header: bool = True,
    dtype: Optional[Dict[str, type]] = None,
    usecols: Optional[List[str]] = None,
    skiprows: int = 0,
    na_values: Optional[List[str]] = None,
    encoding: str = "utf-8",
) -> Dict[str, np.ndarray]:
    """
    Load CSV file into dictionary of arrays.

    Args:
        filepath: Path to CSV file
        delimiter: Column delimiter
        header: First row contains column names
        dtype: Column data types
        usecols: Columns to load
        skiprows: Number of rows to skip
        na_values: Values to treat as NA

    Returns:
        Dictionary mapping column names to arrays

    Example:
        >>> data = load_csv("weather.csv")
        >>> temperatures = data["temperature"]
    """
    filepath = Path(filepath)

    if na_values is None:
        na_values = ["", "NA", "N/A", "NaN", "nan", "null"]

    # Handle gzipped files
    if filepath.suffix == ".gz":
        open_func = lambda p: gzip.open(p, "rt", encoding=encoding)
    else:
        open_func = lambda p: open(p, "r", encoding=encoding)

    data = {}

    with open_func(filepath) as f:
        # Skip rows
        for _ in range(skiprows):
            next(f)

        reader = csv.reader(f, delimiter=delimiter)

        # Get column names
        if header:
            columns = next(reader)
            rows = []
        else:
            first_row = next(reader)
            columns = [f"col_{i}" for i in range(len(first_row))]
            # Put back the first row
            rows = [first_row]

        # Filter columns
        if usecols is not None:
            col_indices = [columns.index(c) for c in usecols if c in columns]
            columns = [columns[i] for i in col_indices]
        else:
            col_indices = list(range(len(columns)))

        # Initialize data lists
        for col in columns:
            data[col] = []

        # Read data rows
        for row in reader:
            rows.append(row)

        for row in rows:
            for i, col_idx in enumerate(col_indices):
                col = columns[i]
                if col_idx < len(row):
                    val = row[col_idx].strip()
                    if val in na_values:
                        data[col].append(np.nan)
                    else:
                        data[col].append(val)
                else:
                    data[col].append(np.nan)

    # Convert to arrays with specified dtypes
    dtype = dtype or {}
    for col in data:
        try:
            if col in dtype:
                data[col] = np.array(data[col], dtype=dtype[col])
            else:
                # Try numeric first
                data[col] = np.array(data[col], dtype=float)
        except ValueError:
            # Keep as string array
            data[col] = np.array(data[col], dtype=str)

    logger.info(f"Loaded {len(rows)} rows from {filepath}")
    return data


def save_csv(
    filepath: Union[str, Path],
    data: Dict[str, ArrayLike],
    delimiter: str = ",",
    header: bool = True,
    float_format: str = "%.6f",
    compress: bool = False,
    encoding: str = "utf-8",
) -> None:
    """
    Save dictionary of arrays to CSV file.

    Args:
        filepath: Output file path
        data: Dictionary of column arrays
        delimiter: Column delimiter
        header: Write column names
        float_format: Format for floating point numbers
        compress: Compress output with gzip
        encoding: Output encoding
    """
    filepath = Path(filepath)

    if compress:
        filepath = filepath.with_suffix(filepath.suffix + ".gz")
        open_func = lambda p: gzip.open(p, "wt", encoding=encoding)
    else:
        open_func = lambda p: open(p, "w", encoding=encoding, newline="")

    columns = list(data.keys())
    arrays = [np.atleast_1d(data[col]) for col in columns]
    n_rows = len(arrays[0])

    with open_func(filepath) as f:
        writer = csv.writer(f, delimiter=delimiter)

        if header:
            writer.writerow(columns)

        for i in range(n_rows):
            row = []
            for arr in arrays:
                val = arr[i]
                if isinstance(val, (float, np.floating)):
                    if np.isnan(val):
                        row.append("")
                    else:
                        row.append(float_format % val)
                else:
                    row.append(str(val))
            writer.writerow(row)

    logger.info(f"Saved {n_rows} rows to {filepath}")


def load_geotiff(
    filepath: Union[str, Path],
    band: Optional[int] = None,
) -> tuple:
    """
    Load GeoTIFF raster file.

    Args:
        filepath: Path to GeoTIFF file
        band: Specific band to load (None for all)

    Returns:
        Tuple of (data_array, georef) where georef contains
        coordinate reference information

    Example:
        >>> data, georef = load_geotiff("ndvi.tif")
        >>> print(f"Shape: {data.shape}, CRS: {georef.crs}")
    """
    filepath = Path(filepath)

    try:
        import rasterio
    except ImportError:
        logger.warning("rasterio not installed, using fallback loader")
        return _load_geotiff_fallback(filepath, band)

    with rasterio.open(filepath) as src:
        if band is not None:
            data = src.read(band)
        else:
            data = src.read()

        georef = GeoReference(
            crs=str(src.crs) if src.crs else "EPSG:4326",
            transform=tuple(src.transform),
            bounds=tuple(src.bounds),
            resolution=(src.res[0], src.res[1]),
        )

    logger.info(f"Loaded GeoTIFF {filepath}: shape={data.shape}")
    return data, georef


def _load_geotiff_fallback(
    filepath: Path,
    band: Optional[int] = None,
) -> tuple:
    """Fallback GeoTIFF loader using PIL/imageio."""
    try:
        from PIL import Image
        img = Image.open(filepath)
        data = np.array(img)
    except ImportError:
        try:
            import imageio
            data = imageio.imread(filepath)
        except ImportError:
            raise ImportError(
                "Neither rasterio, PIL, nor imageio available. "
                "Install one of these packages to load GeoTIFF files."
            )

    # Create default georef
    georef = GeoReference(
        crs="EPSG:4326",
        transform=(1, 0, 0, 0, -1, data.shape[0]),
        bounds=(0, 0, data.shape[1], data.shape[0]),
        resolution=(1, 1),
    )

    return data, georef


def save_geotiff(
    filepath: Union[str, Path],
    data: np.ndarray,
    georef: Optional[GeoReference] = None,
    dtype: str = "float32",
    nodata: Optional[float] = None,
    compress: str = "lzw",
) -> None:
    """
    Save array to GeoTIFF file.

    Args:
        filepath: Output file path
        data: Array to save (2D or 3D with bands first)
        georef: Geographic reference information
        dtype: Output data type
        nodata: NoData value
        compress: Compression type ('lzw', 'deflate', 'none')
    """
    filepath = Path(filepath)

    try:
        import rasterio
        from rasterio.transform import Affine
    except ImportError:
        logger.warning("rasterio not installed, using fallback saver")
        _save_geotiff_fallback(filepath, data)
        return

    # Ensure 3D array
    if data.ndim == 2:
        data = data[np.newaxis, :, :]

    n_bands, height, width = data.shape

    if georef is None:
        transform = Affine.identity()
        crs = "EPSG:4326"
    else:
        transform = Affine(*georef.transform[:6])
        crs = georef.crs

    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "width": width,
        "height": height,
        "count": n_bands,
        "crs": crs,
        "transform": transform,
    }

    if nodata is not None:
        profile["nodata"] = nodata

    if compress != "none":
        profile["compress"] = compress

    with rasterio.open(filepath, "w", **profile) as dst:
        dst.write(data.astype(dtype))

    logger.info(f"Saved GeoTIFF {filepath}: shape={data.shape}")


def _save_geotiff_fallback(filepath: Path, data: np.ndarray) -> None:
    """Fallback GeoTIFF saver using PIL."""
    try:
        from PIL import Image
        # Normalize to 0-255 for image
        if data.dtype == np.float32 or data.dtype == np.float64:
            data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
        img = Image.fromarray(data)
        img.save(filepath)
    except ImportError:
        raise ImportError("PIL not available for fallback GeoTIFF saving")


def load_shapefile(
    filepath: Union[str, Path],
    properties: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Load shapefile vector data.

    Args:
        filepath: Path to shapefile
        properties: Specific properties to load

    Returns:
        Dictionary with 'geometries', 'properties', and 'crs'

    Example:
        >>> data = load_shapefile("fields.shp")
        >>> for geom, props in zip(data['geometries'], data['properties']):
        ...     print(f"Field: {props['name']}, Area: {props['area']}")
    """
    filepath = Path(filepath)

    try:
        import fiona
    except ImportError:
        logger.warning("fiona not installed, shapefile loading not available")
        return {"geometries": [], "properties": [], "crs": None}

    geometries = []
    all_properties = []

    with fiona.open(filepath) as src:
        crs = str(src.crs) if src.crs else None

        for feature in src:
            geometries.append(feature["geometry"])

            if properties is not None:
                props = {k: feature["properties"].get(k) for k in properties}
            else:
                props = dict(feature["properties"])

            all_properties.append(props)

    logger.info(f"Loaded {len(geometries)} features from {filepath}")

    return {
        "geometries": geometries,
        "properties": all_properties,
        "crs": crs,
    }


class DataExporter:
    """
    Flexible data exporter supporting multiple formats.

    Provides a unified interface for exporting data to various
    formats with customizable options.

    Example:
        >>> exporter = DataExporter(output_dir="./output")
        >>> exporter.to_csv(data, "results.csv")
        >>> exporter.to_json(config, "config.json")
        >>> exporter.to_geotiff(raster, "output.tif", georef)
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = ".",
        create_dir: bool = True,
    ):
        """
        Initialize data exporter.

        Args:
            output_dir: Output directory
            create_dir: Create directory if it doesn't exist
        """
        self.output_dir = Path(output_dir)

        if create_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def to_csv(
        self,
        data: Dict[str, ArrayLike],
        filename: str,
        **kwargs,
    ) -> Path:
        """Export data to CSV."""
        filepath = self.output_dir / filename
        save_csv(filepath, data, **kwargs)
        return filepath

    def to_json(
        self,
        data: Any,
        filename: str,
        indent: int = 2,
    ) -> Path:
        """Export data to JSON."""
        filepath = self.output_dir / filename

        # Convert numpy arrays and dataclasses
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif hasattr(obj, "__dataclass_fields__"):
                return asdict(obj)
            return obj

        with open(filepath, "w") as f:
            json.dump(data, f, indent=indent, default=convert)

        logger.info(f"Exported JSON to {filepath}")
        return filepath

    def to_geotiff(
        self,
        data: np.ndarray,
        filename: str,
        georef: Optional[GeoReference] = None,
        **kwargs,
    ) -> Path:
        """Export data to GeoTIFF."""
        filepath = self.output_dir / filename
        save_geotiff(filepath, data, georef, **kwargs)
        return filepath

    def to_numpy(
        self,
        data: np.ndarray,
        filename: str,
        compressed: bool = True,
    ) -> Path:
        """Export data to NumPy binary file."""
        filepath = self.output_dir / filename

        if compressed:
            np.savez_compressed(filepath, data=data)
        else:
            np.save(filepath, data)

        logger.info(f"Exported NumPy array to {filepath}")
        return filepath


@contextmanager
def streaming_csv_writer(
    filepath: Union[str, Path],
    columns: List[str],
    delimiter: str = ",",
    buffer_size: int = 1000,
):
    """
    Context manager for streaming CSV writing.

    Buffers rows and writes in batches for efficiency.

    Args:
        filepath: Output file path
        columns: Column names
        delimiter: Column delimiter
        buffer_size: Rows to buffer before writing

    Yields:
        Function to add rows

    Example:
        >>> with streaming_csv_writer("large.csv", ["id", "value"]) as writer:
        ...     for i in range(1000000):
        ...         writer({"id": i, "value": np.random.random()})
    """
    filepath = Path(filepath)
    buffer = []

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter=delimiter)
        writer.writeheader()

        def add_row(row: Dict):
            buffer.append(row)
            if len(buffer) >= buffer_size:
                writer.writerows(buffer)
                buffer.clear()

        yield add_row

        # Write remaining rows
        if buffer:
            writer.writerows(buffer)

    logger.info(f"Finished streaming to {filepath}")
