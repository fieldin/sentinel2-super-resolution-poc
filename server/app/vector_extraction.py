"""
Vector Extraction Module - Field Boundary Polygon Generation

Extracts field boundary polygons from Sentinel-2 imagery using classical
segmentation approaches. Designed to work with both NDVI-capable multi-band
imagery and RGB-only super-resolution outputs.

Key Features:
- Processes entire AOI at once (no tile-based splitting)
- NDVI-based segmentation when spectral bands available
- HSV green mask fallback for RGB-only images
- Topology cleanup with buffer(0), sliver removal, simplification
- Outputs GeoJSON with attributes: id, area_ha, source, confidence, created_at

Author: Vector Intelligence POC
"""

import json
import subprocess
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.mask import mask as rasterio_mask
from rasterio.features import shapes as rasterio_shapes
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy import ndimage
from scipy.ndimage import binary_fill_holes, binary_opening, binary_closing
import cv2

# Shapely imports for geometry processing
try:
    from shapely.geometry import shape, mapping, Polygon, MultiPolygon
    from shapely.ops import unary_union
    from shapely.validation import make_valid
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

from .utils import setup_logging, ensure_directory, write_json, read_json

logger = setup_logging("vector_extraction")


@dataclass
class ExtractionConfig:
    """Configuration for field polygon extraction."""
    # Smoothing parameters
    gaussian_sigma: float = 2.0  # Gaussian blur sigma
    
    # Segmentation thresholds
    ndvi_threshold: float = 0.3  # NDVI threshold for vegetation
    hsv_green_hue_range: Tuple[int, int] = (35, 85)  # HSV hue range for green
    hsv_saturation_min: int = 30  # Min saturation for green detection
    hsv_value_min: int = 30  # Min value for green detection
    
    # Morphological operations
    morph_kernel_size: int = 5  # Kernel size for morphological ops
    morph_iterations: int = 2  # Iterations for open/close
    
    # Polygon filtering
    min_area_ha: float = 0.1  # Minimum polygon area in hectares
    max_area_ha: float = 500.0  # Maximum polygon area in hectares
    simplify_tolerance_m: float = 5.0  # Simplification tolerance in meters
    
    # Processing limits
    max_pixels_for_full_res: int = 50_000_000  # Max pixels before downsampling
    downsample_factor: int = 2  # Downsample factor for large images


def load_aoi_geojson(aoi_path: Path) -> dict:
    """
    Load AOI from GeoJSON file.
    
    Args:
        aoi_path: Path to GeoJSON file
        
    Returns:
        GeoJSON geometry object
    """
    logger.info(f"Loading AOI from: {aoi_path}")
    data = read_json(aoi_path)
    
    # Handle FeatureCollection
    if data.get("type") == "FeatureCollection":
        features = data.get("features", [])
        if not features:
            raise ValueError("No features found in AOI GeoJSON")
        # Return the first feature's geometry
        return features[0]["geometry"]
    elif data.get("type") == "Feature":
        return data["geometry"]
    else:
        # Assume it's a raw geometry
        return data


def get_raster_bounds_and_crs(raster_path: Path) -> Tuple[list, str]:
    """
    Get raster bounds and CRS.
    
    Args:
        raster_path: Path to raster file
        
    Returns:
        Tuple of (bounds as [west, south, east, north], CRS string)
    """
    with rasterio.open(raster_path) as src:
        bounds = list(src.bounds)
        crs = src.crs.to_string() if src.crs else "EPSG:4326"
    return bounds, crs


def crop_raster_to_aoi(
    raster_path: Path,
    aoi_geom: dict,
    output_path: Path
) -> Path:
    """
    Crop raster to AOI boundary using rasterio.
    
    Args:
        raster_path: Input raster path
        aoi_geom: AOI geometry in EPSG:4326
        output_path: Output cropped raster path
        
    Returns:
        Path to cropped raster
    """
    logger.info(f"Cropping raster to AOI: {raster_path}")
    
    with rasterio.open(raster_path) as src:
        # Reproject AOI to raster CRS if needed
        aoi_shapes = [aoi_geom]
        
        try:
            out_image, out_transform = rasterio_mask(
                src,
                aoi_shapes,
                crop=True,
                filled=True,
                nodata=0
            )
        except Exception as e:
            logger.warning(f"Crop failed, using full raster: {e}")
            return raster_path
        
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "compress": "lzw"
        })
        
        ensure_directory(output_path.parent)
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)
    
    logger.info(f"Cropped raster saved: {output_path}")
    return output_path


def compute_ndvi(raster_path: Path) -> Optional[np.ndarray]:
    """
    Compute NDVI from raster if NIR and Red bands are available.
    
    Assumes standard Sentinel-2 band ordering:
    - Band 4: Red (around 665nm)
    - Band 8: NIR (around 842nm)
    
    For RGB images, returns None.
    
    Args:
        raster_path: Path to raster file
        
    Returns:
        NDVI array (values -1 to 1) or None if not enough bands
    """
    with rasterio.open(raster_path) as src:
        band_count = src.count
        logger.info(f"Raster has {band_count} bands")
        
        if band_count < 4:
            logger.info("Not enough bands for NDVI, will use RGB fallback")
            return None
        
        # Try to read Red (band 4) and NIR (band 8 or 5 depending on image)
        try:
            # Standard Sentinel-2 ordering
            if band_count >= 8:
                red = src.read(4).astype(np.float32)
                nir = src.read(8).astype(np.float32)
            elif band_count >= 5:
                # Some processed images might have fewer bands
                red = src.read(3).astype(np.float32)
                nir = src.read(4).astype(np.float32)
            else:
                return None
            
            # Compute NDVI
            denominator = nir + red
            ndvi = np.where(
                denominator > 0,
                (nir - red) / denominator,
                0
            )
            
            logger.info(f"NDVI computed: min={ndvi.min():.3f}, max={ndvi.max():.3f}")
            return ndvi
            
        except Exception as e:
            logger.warning(f"Failed to compute NDVI: {e}")
            return None


def compute_green_mask_hsv(raster_path: Path, config: ExtractionConfig) -> np.ndarray:
    """
    Compute vegetation mask using HSV color space from RGB image.
    
    This is a fallback when NDVI is not available (RGB-only images).
    
    Args:
        raster_path: Path to RGB raster
        config: Extraction configuration
        
    Returns:
        Binary mask (0 or 1) indicating vegetation
    """
    logger.info("Computing green mask using HSV color space")
    
    with rasterio.open(raster_path) as src:
        # Read RGB bands (assuming first 3 bands)
        rgb = np.dstack([
            src.read(1),
            src.read(2),
            src.read(3)
        ])
        
        # Normalize to 0-255 if needed
        if rgb.max() > 255:
            rgb = (rgb / rgb.max() * 255).astype(np.uint8)
        else:
            rgb = rgb.astype(np.uint8)
    
    # Convert to HSV
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    
    # Create mask for green vegetation
    hue_min, hue_max = config.hsv_green_hue_range
    lower_green = np.array([hue_min, config.hsv_saturation_min, config.hsv_value_min])
    upper_green = np.array([hue_max, 255, 255])
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Also detect brownish vegetation (dry crops)
    lower_brown = np.array([10, 20, 40])
    upper_brown = np.array([35, 200, 200])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # Combine masks
    combined = cv2.bitwise_or(mask, mask_brown)
    
    logger.info(f"HSV green mask: {np.sum(combined > 0)} vegetation pixels")
    return (combined > 0).astype(np.float32)


def compute_vegetation_mask(
    raster_path: Path,
    config: ExtractionConfig
) -> Tuple[np.ndarray, str]:
    """
    Compute vegetation mask using NDVI or HSV fallback.
    
    Args:
        raster_path: Path to raster file
        config: Extraction configuration
        
    Returns:
        Tuple of (binary mask, source type "ndvi" or "rgb_fallback")
    """
    # Try NDVI first
    ndvi = compute_ndvi(raster_path)
    
    if ndvi is not None:
        # Threshold NDVI to create binary mask
        mask = (ndvi > config.ndvi_threshold).astype(np.float32)
        source = "ndvi"
        logger.info(f"Using NDVI-based mask (threshold={config.ndvi_threshold})")
    else:
        # Fallback to HSV green detection
        mask = compute_green_mask_hsv(raster_path, config)
        source = "rgb_fallback"
        logger.info("Using HSV color-based mask (RGB fallback)")
    
    return mask, source


def segment_fields(
    vegetation_mask: np.ndarray,
    config: ExtractionConfig
) -> np.ndarray:
    """
    Segment vegetation mask into distinct field regions.
    
    Uses classical image processing:
    1. Gaussian smoothing
    2. Morphological operations (close holes, remove noise)
    3. Watershed or connected components
    
    Args:
        vegetation_mask: Binary vegetation mask
        config: Extraction configuration
        
    Returns:
        Labeled array where each field has a unique integer label
    """
    logger.info("Segmenting fields from vegetation mask")
    
    # Step 1: Gaussian smoothing
    smoothed = ndimage.gaussian_filter(vegetation_mask, sigma=config.gaussian_sigma)
    binary = (smoothed > 0.5).astype(np.uint8)
    
    # Step 2: Morphological cleanup
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.morph_kernel_size, config.morph_kernel_size)
    )
    
    # Close small gaps within fields
    closed = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=config.morph_iterations
    )
    
    # Open to remove small noise
    opened = cv2.morphologyEx(
        closed,
        cv2.MORPH_OPEN,
        kernel,
        iterations=config.morph_iterations
    )
    
    # Fill holes in binary mask
    filled = binary_fill_holes(opened).astype(np.uint8)
    
    # Step 3: Edge detection for watershed markers
    # Compute distance transform
    dist = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    
    # Find local maxima as field centers
    local_max_mask = (dist > 0.3 * dist.max()).astype(np.uint8)
    
    # Connected components for markers
    _, markers = cv2.connectedComponents(local_max_mask)
    markers = markers + 1  # Background is 1, not 0
    markers[filled == 0] = 0  # Unknown areas
    
    # Apply watershed
    # Create 3-channel image for watershed
    img_3ch = cv2.cvtColor(
        (filled * 255).astype(np.uint8),
        cv2.COLOR_GRAY2BGR
    )
    
    markers_ws = cv2.watershed(img_3ch, markers.astype(np.int32))
    
    # Clean up watershed result
    # -1 are boundaries, set to 0
    markers_ws[markers_ws == -1] = 0
    markers_ws[markers_ws == 1] = 0  # Background
    
    num_fields = len(np.unique(markers_ws)) - 1  # Exclude 0
    logger.info(f"Segmented {num_fields} potential field regions")
    
    return markers_ws


def labels_to_polygons(
    labels: np.ndarray,
    transform,
    crs: str,
    config: ExtractionConfig
) -> List[dict]:
    """
    Convert labeled raster to polygon features.
    
    Args:
        labels: Labeled array from segmentation
        transform: Rasterio affine transform
        crs: Coordinate reference system
        config: Extraction configuration
        
    Returns:
        List of GeoJSON feature dictionaries
    """
    logger.info("Converting labels to polygons")
    
    if not SHAPELY_AVAILABLE:
        raise ImportError("shapely is required for polygon processing")
    
    features = []
    
    # Get unique labels (excluding 0 = background)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]
    
    logger.info(f"Processing {len(unique_labels)} labeled regions")
    
    for label_val in unique_labels:
        # Create binary mask for this label
        mask = (labels == label_val).astype(np.uint8)
        
        # Extract shapes
        shapes_gen = rasterio_shapes(
            mask,
            mask=mask,
            transform=transform,
            connectivity=8
        )
        
        for geom, value in shapes_gen:
            if value == 0:
                continue
                
            # Convert to shapely geometry
            try:
                poly = shape(geom)
                
                # Make geometry valid
                if not poly.is_valid:
                    poly = make_valid(poly)
                
                # Apply buffer(0) to clean topology
                poly = poly.buffer(0)
                
                if poly.is_empty:
                    continue
                
                features.append({
                    "geometry": poly,
                    "label": int(label_val)
                })
                
            except Exception as e:
                logger.warning(f"Failed to process region {label_val}: {e}")
                continue
    
    logger.info(f"Extracted {len(features)} raw polygons")
    return features


def cleanup_polygons(
    features: List[dict],
    config: ExtractionConfig,
    pixel_size_m: float = 10.0
) -> List[dict]:
    """
    Clean up and filter polygons.
    
    - Remove slivers by area threshold
    - Simplify boundaries
    - Ensure no overlaps
    - Add attributes
    
    Args:
        features: List of feature dicts with 'geometry' key
        config: Extraction configuration
        pixel_size_m: Approximate pixel size in meters
        
    Returns:
        Cleaned list of feature dictionaries
    """
    logger.info("Cleaning up polygons")
    
    if not features:
        return []
    
    cleaned = []
    
    # Convert area thresholds from hectares to square meters
    min_area_m2 = config.min_area_ha * 10000
    max_area_m2 = config.max_area_ha * 10000
    
    # Simplification tolerance (convert from meters to degrees approximately)
    # At equator, 1 degree ≈ 111km, so 1m ≈ 0.00001 degrees
    simplify_tolerance = config.simplify_tolerance_m * 0.00001
    
    for feat in features:
        poly = feat["geometry"]
        
        # Skip invalid geometries
        if poly is None or poly.is_empty:
            continue
        
        # Handle MultiPolygon by taking largest
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda g: g.area)
        
        # Simplify
        poly = poly.simplify(simplify_tolerance, preserve_topology=True)
        
        # Calculate area (approximate, in degrees^2)
        # Convert to approximate m^2 for filtering
        # At ~37°N latitude, 1 degree ≈ 87km longitude, 111km latitude
        area_deg2 = poly.area
        area_m2_approx = area_deg2 * (87000 * 111000)  # Very rough approximation
        
        # Filter by area
        if area_m2_approx < min_area_m2:
            continue
        if area_m2_approx > max_area_m2:
            continue
        
        # Convert area to hectares
        area_ha = area_m2_approx / 10000
        
        cleaned.append({
            "geometry": poly,
            "label": feat["label"],
            "area_ha": round(area_ha, 2)
        })
    
    logger.info(f"After cleanup: {len(cleaned)} polygons (filtered {len(features) - len(cleaned)})")
    return cleaned


def compute_confidence(
    feature: dict,
    source: str,
    total_features: int
) -> float:
    """
    Compute a heuristic confidence score for a field polygon.
    
    Based on:
    - Area (medium-sized fields are more likely correct)
    - Source method (NDVI is more reliable than RGB)
    - Shape regularity (more regular = higher confidence)
    
    Args:
        feature: Feature dictionary with geometry
        source: Source method ("ndvi" or "rgb_fallback")
        total_features: Total number of features extracted
        
    Returns:
        Confidence score between 0 and 1
    """
    poly = feature["geometry"]
    area_ha = feature.get("area_ha", 1.0)
    
    # Base confidence from source
    if source == "ndvi":
        base_conf = 0.7
    else:
        base_conf = 0.5
    
    # Area factor (prefer 1-50 ha fields)
    if 1.0 <= area_ha <= 50.0:
        area_factor = 1.0
    elif 0.5 <= area_ha < 1.0 or 50.0 < area_ha <= 100.0:
        area_factor = 0.8
    else:
        area_factor = 0.6
    
    # Compactness factor (circle = 1, elongated = lower)
    try:
        perimeter = poly.length
        area = poly.area
        if perimeter > 0:
            compactness = 4 * np.pi * area / (perimeter ** 2)
            compactness = min(compactness, 1.0)
        else:
            compactness = 0.5
    except:
        compactness = 0.5
    
    # Shape factor (prefer more compact shapes)
    shape_factor = 0.7 + 0.3 * compactness
    
    # Combine factors
    confidence = base_conf * area_factor * shape_factor
    
    return round(min(max(confidence, 0.1), 1.0), 2)


def create_geojson_output(
    features: List[dict],
    source: str,
    output_path: Path
) -> dict:
    """
    Create GeoJSON FeatureCollection from processed polygons.
    
    Args:
        features: List of cleaned feature dictionaries
        source: Source method used
        output_path: Output file path
        
    Returns:
        GeoJSON dictionary
    """
    logger.info(f"Creating GeoJSON with {len(features)} features")
    
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    geojson_features = []
    
    for i, feat in enumerate(features):
        poly = feat["geometry"]
        
        # Compute confidence
        confidence = compute_confidence(feat, source, len(features))
        
        # Generate stable ID from geometry
        geom_str = str(poly.wkt)[:100]
        feature_id = hashlib.md5(geom_str.encode()).hexdigest()[:12]
        
        geojson_features.append({
            "type": "Feature",
            "id": feature_id,
            "properties": {
                "id": feature_id,
                "field_index": i + 1,
                "area_ha": feat["area_ha"],
                "source": source,
                "confidence": confidence,
                "created_at": timestamp
            },
            "geometry": mapping(poly)
        })
    
    geojson = {
        "type": "FeatureCollection",
        "properties": {
            "generated_at": timestamp,
            "source_method": source,
            "feature_count": len(geojson_features),
            "version": "1.0"
        },
        "features": geojson_features
    }
    
    # Write to file
    ensure_directory(output_path.parent)
    write_json(geojson, output_path)
    
    logger.info(f"GeoJSON saved: {output_path}")
    return geojson


def extract_field_polygons(
    aoi_geojson: Path,
    raster_paths: List[Path],
    out_dir: Path,
    config: Optional[ExtractionConfig] = None
) -> dict:
    """
    Main entry point for field polygon extraction.
    
    Processes the entire AOI at once to ensure polygons don't break across tiles.
    
    Args:
        aoi_geojson: Path to AOI GeoJSON file
        raster_paths: List of raster file paths (will use first valid one)
        out_dir: Output directory for results
        config: Extraction configuration (uses defaults if None)
        
    Returns:
        Dictionary with extraction results and metadata
    """
    logger.info("=" * 60)
    logger.info("FIELD POLYGON EXTRACTION")
    logger.info("=" * 60)
    
    if config is None:
        config = ExtractionConfig()
    
    # Ensure shapely is available
    if not SHAPELY_AVAILABLE:
        raise ImportError(
            "shapely is required for vector extraction. "
            "Install with: pip install shapely"
        )
    
    # Load AOI
    aoi_geom = load_aoi_geojson(aoi_geojson)
    logger.info(f"AOI type: {aoi_geom.get('type', 'unknown')}")
    
    # Find first valid raster
    raster_path = None
    for path in raster_paths:
        path = Path(path)
        if path.exists():
            raster_path = path
            break
    
    if raster_path is None:
        raise FileNotFoundError(f"No valid raster files found in: {raster_paths}")
    
    logger.info(f"Using raster: {raster_path}")
    
    # Get raster info
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs.to_string() if src.crs else "EPSG:4326"
        raster_transform = src.transform
        raster_shape = (src.height, src.width)
        pixel_size_m = abs(src.transform[0]) * 111000  # Rough conversion for degrees
    
    logger.info(f"Raster CRS: {raster_crs}")
    logger.info(f"Raster shape: {raster_shape}")
    logger.info(f"Approx pixel size: {pixel_size_m:.1f}m")
    
    # Check if downsampling is needed
    total_pixels = raster_shape[0] * raster_shape[1]
    if total_pixels > config.max_pixels_for_full_res:
        logger.warning(
            f"Large raster ({total_pixels:,} pixels). "
            f"Consider downsampling for faster processing."
        )
    
    # Crop raster to AOI (optional, improves performance)
    temp_dir = out_dir / "temp"
    ensure_directory(temp_dir)
    cropped_path = temp_dir / f"cropped_{raster_path.stem}.tif"
    
    try:
        working_raster = crop_raster_to_aoi(raster_path, aoi_geom, cropped_path)
    except Exception as e:
        logger.warning(f"Crop failed, using original raster: {e}")
        working_raster = raster_path
    
    # Compute vegetation mask
    vegetation_mask, source_method = compute_vegetation_mask(working_raster, config)
    
    # Segment fields
    labels = segment_fields(vegetation_mask, config)
    
    # Get transform for the working raster
    with rasterio.open(working_raster) as src:
        transform = src.transform
        crs = src.crs.to_string() if src.crs else "EPSG:4326"
    
    # Convert labels to polygons
    raw_features = labels_to_polygons(labels, transform, crs, config)
    
    # Cleanup polygons
    cleaned_features = cleanup_polygons(raw_features, config, pixel_size_m)
    
    # Create output GeoJSON
    output_path = out_dir / "fields.geojson"
    geojson = create_geojson_output(cleaned_features, source_method, output_path)
    
    # Create metadata
    metadata = {
        "status": "completed",
        "input_raster": str(raster_path),
        "aoi_geojson": str(aoi_geojson),
        "output_geojson": str(output_path),
        "feature_count": len(cleaned_features),
        "source_method": source_method,
        "config": {
            "ndvi_threshold": config.ndvi_threshold,
            "min_area_ha": config.min_area_ha,
            "max_area_ha": config.max_area_ha,
            "simplify_tolerance_m": config.simplify_tolerance_m
        },
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
    
    metadata_path = out_dir / "extraction_metadata.json"
    write_json(metadata, metadata_path)
    
    # Cleanup temp files
    try:
        if cropped_path.exists() and cropped_path != raster_path:
            cropped_path.unlink()
        if temp_dir.exists():
            temp_dir.rmdir()
    except:
        pass
    
    logger.info("=" * 60)
    logger.info(f"EXTRACTION COMPLETE: {len(cleaned_features)} field polygons")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)
    
    return {
        "status": "completed",
        "output_path": str(output_path),
        "feature_count": len(cleaned_features),
        "source_method": source_method,
        "geojson": geojson
    }


# ============================================================
# CLI Interface
# ============================================================

def main():
    """Command-line interface for vector extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract field boundary polygons from satellite imagery"
    )
    parser.add_argument(
        "--aoi", "-a",
        type=str,
        required=True,
        help="Path to AOI GeoJSON file"
    )
    parser.add_argument(
        "--rasters", "-r",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to input raster file(s)"
    )
    parser.add_argument(
        "--out", "-o",
        type=str,
        required=True,
        help="Output directory for vectors"
    )
    parser.add_argument(
        "--ndvi-threshold",
        type=float,
        default=0.3,
        help="NDVI threshold for vegetation (default: 0.3)"
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.1,
        help="Minimum field area in hectares (default: 0.1)"
    )
    parser.add_argument(
        "--max-area",
        type=float,
        default=500.0,
        help="Maximum field area in hectares (default: 500)"
    )
    parser.add_argument(
        "--simplify",
        type=float,
        default=5.0,
        help="Simplification tolerance in meters (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = ExtractionConfig(
        ndvi_threshold=args.ndvi_threshold,
        min_area_ha=args.min_area,
        max_area_ha=args.max_area,
        simplify_tolerance_m=args.simplify
    )
    
    # Run extraction
    result = extract_field_polygons(
        aoi_geojson=Path(args.aoi),
        raster_paths=[Path(r) for r in args.rasters],
        out_dir=Path(args.out),
        config=config
    )
    
    print(f"\nExtracted {result['feature_count']} field polygons")
    print(f"Output: {result['output_path']}")


if __name__ == "__main__":
    main()

