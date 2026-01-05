"""
Vector Extraction V2 - Enhanced Field Boundary Detection

Improvements over V1:
1. Uses actual Sentinel-2 bands (B04, B08) for NDVI computation
2. Gradient-based segmentation (Sobel → watershed) for better boundary detection
3. OSM road clipping to split fields correctly
4. Rich per-field attributes: mean_ndvi, ndvi_std, edge_strength, confidence
5. Management zones layer (k-means clustering) for sub-field variability

Author: Vector Intelligence POC v2
"""

import json
import hashlib
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field

import numpy as np
import rasterio
from rasterio.features import shapes as rasterio_shapes
from rasterio.mask import mask as rasterio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy import ndimage
from scipy.ndimage import binary_fill_holes, label
import cv2

try:
    from affine import Affine
except Exception:
    Affine = None

try:
    from shapely.geometry import shape, mapping, Polygon, MultiPolygon, LineString, box
    from shapely.ops import unary_union, split
    from shapely.validation import make_valid

    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False

try:
    from sklearn.cluster import KMeans

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .utils import setup_logging, ensure_directory, write_json, read_json

logger = setup_logging("vector_extraction_v2")


@dataclass
class ExtractionConfigV2:
    """Enhanced configuration for field polygon extraction."""

    # NDVI computation
    # For our stacked multiband file we use:
    #   1=B04(red), 2=B08(nir), 3=SCL
    red_band_idx: int = 1  # B04
    nir_band_idx: int = 2  # B08
    ndvi_veg_threshold: float = 0.2  # Min NDVI to consider vegetation

    # Gradient-based segmentation
    gaussian_sigma: float = 1.5  # Smoothing before gradient
    gradient_threshold: float = 0.05  # Min gradient to consider edge
    watershed_compactness: float = 0.01  # Watershed compactness parameter

    # Morphological operations
    morph_kernel_size: int = 3
    min_region_pixels: int = 100  # Minimum region size in pixels

    # Polygon filtering
    min_area_ha: float = 0.5  # Filter out tiny fragments
    max_area_ha: float = 500.0
    simplify_tolerance_m: float = 3.0  # Tighter simplification

    # OSM road clipping
    use_osm_roads: bool = True
    road_buffer_m: float = 5.0  # Buffer around roads

    # Management zones
    generate_zones: bool = True
    num_zones: int = 3  # Number of k-means clusters

    # Multi-date composite (future)
    use_composite: bool = False
    composite_dates: int = 5

    # Performance (POC-friendly)
    # NOTE: Vector extraction is run once per AOI, so we prefer correctness, but for large AOIs
    # we downsample for segmentation to keep runtime reasonable.
    max_pixels_for_full_res: int = 8_000_000
    downsample_factor: int = 4


# ============================================================
# SENTINEL-2 MULTI-BAND FETCHING
# ============================================================


def fetch_sentinel2_bands(
    aoi_geom: dict,
    output_dir: Path,
    max_cloud_cover: float = 20.0,
    max_age_days: int = 30,
) -> Tuple[Optional[Path], dict]:
    """
    Fetch Sentinel-2 L2A with multiple bands (B04, B08, SCL).

    Returns raster with bands: [B04_red, B08_nir, SCL]
    """
    logger.info("Fetching multi-band Sentinel-2 data...")

    stac_url = "https://earth-search.aws.element84.com/v1/search"

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=max_age_days)

    search_payload = {
        "collections": ["sentinel-2-l2a"],
        "intersects": aoi_geom,
        "datetime": f"{start_date.strftime('%Y-%m-%d')}T00:00:00Z/{end_date.strftime('%Y-%m-%d')}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lte": max_cloud_cover}},
        "limit": 5,
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    }

    try:
        response = requests.post(stac_url, json=search_payload, timeout=30)
        response.raise_for_status()
        results = response.json()
    except Exception as e:
        logger.error(f"STAC search failed: {e}")
        return None, {}

    features = results.get("features", [])
    if not features:
        logger.warning("No Sentinel-2 scenes found")
        return None, {}

    scene = features[0]
    assets = scene.get("assets", {})
    props = scene.get("properties", {})

    logger.info(
        f"Selected scene: {scene.get('id')} (cloud: {props.get('eo:cloud_cover', 'N/A')}%)"
    )

    # Get band URLs
    band_urls = {}
    for band_name in ["red", "nir", "scl"]:
        if band_name in assets:
            band_urls[band_name] = assets[band_name].get("href")

    if not band_urls.get("red") or not band_urls.get("nir"):
        logger.warning("Missing required bands (red/nir)")
        return None, {}

    # Download and stack bands
    ensure_directory(output_dir)
    output_path = (
        output_dir / f"s2_multiband_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tif"
    )

    # Use GDAL VRT to stack bands
    import subprocess

    # Get AOI bounds
    from shapely.geometry import shape as shp

    aoi_shape = shp(aoi_geom)
    bounds = aoi_shape.bounds  # (minx, miny, maxx, maxy)

    # Build GDAL command to download and clip
    band_files = []
    for band_name, url in band_urls.items():
        temp_file = output_dir / f"temp_{band_name}.tif"
        cmd = [
            "gdalwarp",
            "-t_srs",
            "EPSG:4326",
            "-te",
            str(bounds[0]),
            str(bounds[1]),
            str(bounds[2]),
            str(bounds[3]),
            "-overwrite",
            f"/vsicurl/{url}",
            str(temp_file),
        ]
        logger.info(f"Downloading {band_name}...")
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
            band_files.append((band_name, temp_file))
        except Exception as e:
            logger.warning(f"Failed to download {band_name}: {e}")

    if len(band_files) < 2:
        logger.error("Could not download required bands")
        return None, {}

    # Stack bands into single file
    logger.info("Stacking bands...")

    # Read first band to get metadata
    with rasterio.open(band_files[0][1]) as src:
        meta = src.meta.copy()
        height, width = src.height, src.width

    meta.update(count=len(band_files), dtype="float32")

    with rasterio.open(output_path, "w", **meta) as dst:
        for i, (band_name, band_file) in enumerate(band_files):
            with rasterio.open(band_file) as src:
                data = src.read(1).astype(np.float32)
                # Normalize to 0-1 range
                if band_name != "scl":
                    data = data / 10000.0  # Sentinel-2 scaling
                dst.write(data, i + 1)

    # Cleanup temp files
    for _, f in band_files:
        try:
            f.unlink()
        except:
            pass

    metadata = {
        "scene_id": scene.get("id"),
        "acquisition_date": props.get("datetime"),
        "cloud_cover": props.get("eo:cloud_cover"),
        "bands": list(band_urls.keys()),
        "output_path": str(output_path),
    }

    logger.info(f"Multi-band raster saved: {output_path}")
    return output_path, metadata


# ============================================================
# NDVI COMPUTATION
# ============================================================


def compute_ndvi_from_bands(
    raster_path: Path, red_band: int = 1, nir_band: int = 2, scl_band: Optional[int] = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute NDVI from multi-band raster.

    Returns:
        Tuple of (ndvi_array, valid_mask)
    """
    logger.info("Computing NDVI from spectral bands...")

    with rasterio.open(raster_path) as src:
        red = src.read(red_band).astype(np.float32)
        nir = src.read(nir_band).astype(np.float32)

        # SCL mask if available
        if scl_band and src.count >= scl_band:
            scl = src.read(scl_band)
            # Valid SCL values (Sentinel-2 L2A):
            # 4=vegetation, 5=bare soil, 6=water, 7=unclassified
            # (we keep these; reject clouds/shadows/snow)
            valid_mask = np.isin(scl.astype(np.int32), [4, 5, 6, 7])
        else:
            valid_mask = (red > 0) & (nir > 0)

    # Compute NDVI
    denominator = nir + red
    ndvi = np.where(denominator > 0, (nir - red) / denominator, 0)

    # Clip to valid range
    ndvi = np.clip(ndvi, -1, 1)

    # Keep this log compact; avoid printing huge arrays/extra text
    logger.info(
        f"NDVI: min={float(ndvi.min()):.3f}, max={float(ndvi.max()):.3f}, "
        f"mean={float(ndvi[valid_mask].mean()) if np.any(valid_mask) else 0.0:.3f}"
    )

    return ndvi, valid_mask


def compute_ndvi_from_rgb(raster_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fallback: Estimate vegetation index from RGB using VARI (Visible Atmospherically
    Resistant Index) which works better for RGB-only imagery.

    VARI = (G - R) / (G + R - B)
    Also compute ExG as secondary: ExG = 2*G - R - B (normalized)
    """
    logger.info("Computing vegetation index from RGB (fallback)...")

    with rasterio.open(raster_path) as src:
        if src.count < 3:
            raise ValueError("Need at least 3 bands for RGB")

        r = src.read(1).astype(np.float32)
        g = src.read(2).astype(np.float32)
        b = src.read(3).astype(np.float32)

    # Normalize to 0-1 range if values are in 0-255
    if r.max() > 1:
        r = r / 255.0
        g = g / 255.0
        b = b / 255.0

    # VARI: Visible Atmospherically Resistant Index
    # Better for RGB-only vegetation detection
    denom = g + r - b
    denom = np.where(np.abs(denom) > 0.01, denom, 0.01)
    vari = (g - r) / denom

    # Also compute ExG for comparison
    total = r + g + b
    total = np.where(total > 0.01, total, 0.01)
    r_norm = r / total
    g_norm = g / total
    b_norm = b / total
    exg = 2 * g_norm - r_norm - b_norm

    # Combine VARI and ExG (weighted average)
    # VARI is more sensitive, ExG is more stable
    veg_index = 0.6 * np.clip(vari, -1, 1) + 0.4 * np.clip(exg * 2, -1, 1)
    veg_index = np.clip(veg_index, -1, 1)

    # Valid mask: exclude very dark and very bright pixels
    brightness = (r + g + b) / 3
    valid_mask = (brightness > 0.05) & (brightness < 0.95)

    logger.info(
        f"Vegetation index (VARI+ExG): min={veg_index[valid_mask].min():.3f}, "
        f"max={veg_index[valid_mask].max():.3f}, mean={veg_index[valid_mask].mean():.3f}"
    )

    return veg_index, valid_mask


# ============================================================
# GRADIENT-BASED SEGMENTATION
# ============================================================


def segment_by_gradient(
    ndvi: np.ndarray, valid_mask: np.ndarray, config: ExtractionConfigV2
) -> np.ndarray:
    """
    Segment fields using gradient-based watershed.

    This detects field BOUNDARIES instead of just vegetation areas.

    Pipeline:
    1. Smooth NDVI
    2. Compute gradient magnitude (Sobel)
    3. Create markers (seeds) in homogeneous regions
    4. Run watershed on gradient
    """
    logger.info("Running gradient-based segmentation...")

    # Step 1: Smooth NDVI
    ndvi_smooth = ndimage.gaussian_filter(ndvi, sigma=config.gaussian_sigma)

    # Step 2: Compute gradient magnitude
    grad_x = ndimage.sobel(ndvi_smooth, axis=1)
    grad_y = ndimage.sobel(ndvi_smooth, axis=0)
    gradient = np.sqrt(grad_x**2 + grad_y**2)

    logger.info(f"Gradient: max={gradient.max():.4f}, mean={gradient.mean():.4f}")

    # Step 3: Create markers
    # Find local minima in gradient (homogeneous regions)
    # These become seeds for watershed

    # Threshold gradient to find likely field interiors
    interior_mask = gradient < config.gradient_threshold
    interior_mask = interior_mask & valid_mask

    # Distance transform to find region centers
    dist = ndimage.distance_transform_edt(interior_mask)

    # Find local maxima in distance (furthest from edges)
    from scipy.ndimage import maximum_filter

    local_max = (dist == maximum_filter(dist, size=20)) & (dist > 5)

    # Label connected components as markers
    markers, num_markers = label(local_max)
    logger.info(f"Found {num_markers} seed markers")

    # Add background marker
    markers[~valid_mask] = num_markers + 1

    # Step 4: Watershed on gradient
    # Convert to format expected by cv2.watershed
    gradient_uint8 = (gradient / gradient.max() * 255).astype(np.uint8)
    gradient_3ch = cv2.cvtColor(gradient_uint8, cv2.COLOR_GRAY2BGR)

    markers_int32 = markers.astype(np.int32)
    cv2.watershed(gradient_3ch, markers_int32)

    # Clean up result
    # -1 are boundaries, set to 0
    markers_int32[markers_int32 == -1] = 0
    # Remove background marker
    markers_int32[markers_int32 == num_markers + 1] = 0

    # Remove small regions
    for label_id in np.unique(markers_int32):
        if label_id == 0:
            continue
        region_size = np.sum(markers_int32 == label_id)
        if region_size < config.min_region_pixels:
            markers_int32[markers_int32 == label_id] = 0

    num_regions = len(np.unique(markers_int32)) - 1
    logger.info(f"Segmented {num_regions} field regions")

    return markers_int32


# ============================================================
# OSM ROAD FETCHING AND CLIPPING
# ============================================================


def fetch_osm_roads(bounds: Tuple[float, float, float, float]) -> List[dict]:
    """
    Fetch roads from OpenStreetMap Overpass API.

    Args:
        bounds: (west, south, east, north) in EPSG:4326

    Returns:
        List of road geometries as GeoJSON
    """
    logger.info("Fetching OSM roads...")

    west, south, east, north = bounds

    # Overpass query for roads
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:30];
    (
      way["highway"~"primary|secondary|tertiary|unclassified|residential|service|track|path"]
        ({south},{west},{north},{east});
    );
    out geom;
    """

    try:
        response = requests.post(overpass_url, data={"data": query}, timeout=60)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.warning(f"OSM fetch failed: {e}")
        return []

    roads = []
    for element in data.get("elements", []):
        if element.get("type") == "way" and "geometry" in element:
            coords = [(p["lon"], p["lat"]) for p in element["geometry"]]
            if len(coords) >= 2:
                roads.append(
                    {
                        "type": "LineString",
                        "coordinates": coords,
                        "properties": {
                            "highway": element.get("tags", {}).get("highway", "unknown")
                        },
                    }
                )

    logger.info(f"Fetched {len(roads)} road segments")
    return roads


def clip_polygons_by_roads(
    polygons: List[Polygon], roads: List[dict], buffer_m: float = 5.0
) -> List[Polygon]:
    """
    Split polygons using buffered road lines.

    This fixes the "two fields separated by a road become one polygon" problem.
    """
    if not roads or not polygons:
        return polygons

    logger.info(f"Clipping {len(polygons)} polygons by {len(roads)} roads...")

    # Convert buffer from meters to degrees (approximate)
    buffer_deg = buffer_m / 111000

    # Create buffered road union
    road_lines = []
    for road in roads:
        try:
            line = shape(road)
            if line.is_valid:
                road_lines.append(line.buffer(buffer_deg))
        except:
            continue

    if not road_lines:
        return polygons

    road_union = unary_union(road_lines)

    # Split each polygon by roads
    result = []
    for poly in polygons:
        try:
            if not poly.is_valid:
                poly = make_valid(poly)

            # Difference removes road areas
            diff = poly.difference(road_union)

            if diff.is_empty:
                continue

            # May result in multiple polygons
            if isinstance(diff, MultiPolygon):
                for p in diff.geoms:
                    if p.area > 0:
                        result.append(p)
            elif isinstance(diff, Polygon) and diff.area > 0:
                result.append(diff)

        except Exception as e:
            logger.warning(f"Road clipping failed for polygon: {e}")
            result.append(poly)

    logger.info(f"After road clipping: {len(result)} polygons")
    return result


# ============================================================
# FIELD STATISTICS
# ============================================================


def compute_field_statistics(
    poly: Polygon,
    ndvi: np.ndarray,
    gradient: np.ndarray,
    transform,
    config: ExtractionConfigV2,
) -> dict:
    """
    Compute rich per-field statistics.

    Returns:
        Dictionary with: mean_ndvi, ndvi_std, p10_ndvi, p90_ndvi, edge_strength, etc.
    """
    from rasterio.features import geometry_mask

    # Create mask for this polygon
    mask = geometry_mask(
        [mapping(poly)], out_shape=ndvi.shape, transform=transform, invert=True
    )

    # Get NDVI values inside polygon
    ndvi_values = ndvi[mask]

    if len(ndvi_values) == 0:
        return {}

    # NDVI statistics
    stats = {
        "mean_ndvi": float(np.mean(ndvi_values)),
        "ndvi_std": float(np.std(ndvi_values)),
        "p10_ndvi": float(np.percentile(ndvi_values, 10)),
        "p90_ndvi": float(np.percentile(ndvi_values, 90)),
        "ndvi_range": float(
            np.percentile(ndvi_values, 90) - np.percentile(ndvi_values, 10)
        ),
    }

    # Edge strength (mean gradient at boundary)
    # Dilate mask to get boundary
    boundary_mask = ndimage.binary_dilation(mask, iterations=2) & ~mask
    if np.any(boundary_mask):
        edge_values = gradient[boundary_mask]
        stats["edge_strength"] = float(np.mean(edge_values))
    else:
        stats["edge_strength"] = 0.0

    # Compactness (Polsby-Popper)
    area = poly.area
    perimeter = poly.length
    if perimeter > 0:
        stats["compactness"] = float(4 * np.pi * area / (perimeter**2))
    else:
        stats["compactness"] = 0.0

    return stats


def compute_confidence_v2(stats: dict, source: str) -> float:
    """
    Compute confidence score based on multiple factors.

    Factors:
    - edge_strength: stronger edges = more confident boundary
    - compactness: more regular shapes = higher confidence
    - ndvi_std: lower variance = more homogeneous (higher confidence)
    - source: spectral bands more reliable than RGB
    """
    score = 0.5  # Base score

    # Edge strength factor (0-0.2)
    edge = stats.get("edge_strength", 0)
    score += min(edge * 2, 0.2)

    # Compactness factor (0-0.15)
    compact = stats.get("compactness", 0)
    score += compact * 0.15

    # Homogeneity factor (0-0.15)
    ndvi_std = stats.get("ndvi_std", 0.5)
    if ndvi_std < 0.1:
        score += 0.15
    elif ndvi_std < 0.2:
        score += 0.1
    elif ndvi_std < 0.3:
        score += 0.05

    # Source factor
    if source == "ndvi_bands":
        score += 0.1
    elif source == "rgb_estimated":
        score += 0.0

    return round(min(max(score, 0.1), 1.0), 2)


# ============================================================
# MANAGEMENT ZONES
# ============================================================


def generate_management_zones(
    poly: Polygon, ndvi: np.ndarray, transform, num_zones: int = 3
) -> List[dict]:
    """
    Generate sub-field management zones using k-means on NDVI.

    Returns list of zone polygons with zone_id and mean_ndvi.
    """
    if not SKLEARN_AVAILABLE:
        return []

    from rasterio.features import geometry_mask

    # Create mask for this polygon
    mask = geometry_mask(
        [mapping(poly)], out_shape=ndvi.shape, transform=transform, invert=True
    )

    # Get coordinates and NDVI values
    rows, cols = np.where(mask)
    if len(rows) < num_zones * 10:
        return []  # Not enough pixels

    ndvi_values = ndvi[mask].reshape(-1, 1)

    # K-means clustering
    kmeans = KMeans(n_clusters=num_zones, random_state=42, n_init=10)
    labels = kmeans.fit_predict(ndvi_values)

    # Create zone masks
    zones = []
    zone_raster = np.zeros_like(ndvi, dtype=np.uint8)
    zone_raster[rows, cols] = labels + 1

    # Polygonize each zone
    for zone_id in range(1, num_zones + 1):
        zone_mask = (zone_raster == zone_id).astype(np.uint8)

        # Get zone NDVI stats
        zone_ndvi = ndvi[zone_raster == zone_id]

        # Polygonize
        for geom, value in rasterio_shapes(
            zone_mask, mask=zone_mask, transform=transform
        ):
            if value == 1:
                zone_poly = shape(geom)
                if zone_poly.is_valid and zone_poly.area > 0:
                    zones.append(
                        {
                            "geometry": zone_poly,
                            "zone_id": int(zone_id),
                            "mean_ndvi": float(np.mean(zone_ndvi)),
                            "zone_class": (
                                ["low", "medium", "high"][zone_id - 1]
                                if num_zones == 3
                                else f"zone_{zone_id}"
                            ),
                        }
                    )

    return zones


# ============================================================
# MAIN EXTRACTION PIPELINE V2
# ============================================================


def extract_field_polygons_v2(
    aoi_geojson: Path,
    raster_paths: List[Path],
    out_dir: Path,
    config: Optional[ExtractionConfigV2] = None,
    fetch_multiband: bool = False,
) -> dict:
    """
    Enhanced field polygon extraction with:
    - Proper NDVI from bands
    - Gradient-based segmentation
    - OSM road clipping
    - Rich statistics
    - Management zones
    """
    logger.info("=" * 60)
    logger.info("FIELD POLYGON EXTRACTION V2 (Enhanced)")
    logger.info("=" * 60)

    if config is None:
        config = ExtractionConfigV2()

    if not SHAPELY_AVAILABLE:
        raise ImportError("shapely is required")

    # Load AOI
    aoi_data = read_json(aoi_geojson)
    if aoi_data.get("type") == "FeatureCollection":
        aoi_geom = aoi_data["features"][0]["geometry"]
    elif aoi_data.get("type") == "Feature":
        aoi_geom = aoi_data["geometry"]
    else:
        aoi_geom = aoi_data

    aoi_shape = shape(aoi_geom)
    aoi_bounds = aoi_shape.bounds

    ensure_directory(out_dir)

    # Determine raster source
    raster_path = None
    source_type = "unknown"

    # Option 1: Fetch multi-band Sentinel-2
    if fetch_multiband:
        multiband_dir = out_dir / "multiband"
        raster_path, fetch_meta = fetch_sentinel2_bands(aoi_geom, multiband_dir)
        if raster_path:
            source_type = "ndvi_bands"

    # Option 2: Use provided raster
    if raster_path is None:
        for path in raster_paths:
            if Path(path).exists():
                raster_path = Path(path)
                break

    if raster_path is None:
        raise FileNotFoundError("No valid raster found")

    logger.info(f"Using raster: {raster_path}")

    # Read raster metadata (+ optionally downsample for speed)
    with rasterio.open(raster_path) as src:
        transform = src.transform
        crs = src.crs
        band_count = src.count
        height, width = src.height, src.width

        total_pixels = width * height
        scale = 1
        if total_pixels > config.max_pixels_for_full_res:
            scale = max(2, int(config.downsample_factor))
            logger.warning(
                f"Large raster ({width}x{height}={total_pixels:,} px). "
                f"Downsampling by {scale}x for faster extraction."
            )

        out_h = max(1, height // scale)
        out_w = max(1, width // scale)

        # Detect SCL stack (avoid reading full band 3)
        is_scl_stack = False
        if band_count >= 3:
            b3_sample = src.read(3, out_shape=(min(256, height), min(256, width)))
            if float(np.max(b3_sample)) <= 20:
                is_scl_stack = True

        if scale > 1 and Affine is not None:
            transform = transform * Affine.scale(width / out_w, height / out_h)

        def read_band(b: int, resampling: Resampling) -> np.ndarray:
            return src.read(b, out_shape=(out_h, out_w), resampling=resampling)

        # Compute NDVI / vegetation index at chosen resolution
        if is_scl_stack:
            logger.info("Detected multiband stack (B04+B08+SCL) → real NDVI")
            red = read_band(config.red_band_idx, Resampling.bilinear).astype(np.float32)
            nir = read_band(config.nir_band_idx, Resampling.bilinear).astype(np.float32)
            scl = read_band(3, Resampling.nearest).astype(np.int32)

            valid_mask = np.isin(scl, [4, 5, 6, 7]) & (red > 0) & (nir > 0)
            denom = nir + red
            ndvi = np.zeros_like(red, dtype=np.float32)
            v = denom > 0
            ndvi[v] = (nir[v] - red[v]) / denom[v]
            ndvi = np.clip(ndvi, -1, 1)
            source_type = "ndvi_bands"
        else:
            logger.info("RGB-only raster → estimated vegetation index (NOT true NDVI)")
            if band_count < 3:
                raise ValueError("Need at least 3 bands for RGB fallback")
            r = read_band(1, Resampling.bilinear).astype(np.float32)
            g = read_band(2, Resampling.bilinear).astype(np.float32)
            b = read_band(3, Resampling.bilinear).astype(np.float32)

            # Normalize heuristic (covers both 0..255 and 0..1 inputs)
            if float(np.nanmax(r)) > 1.5:
                r = r / 255.0
                g = g / 255.0
                b = b / 255.0

            denom = g + r - b
            denom = np.where(np.abs(denom) > 0.01, denom, 0.01)
            vari = (g - r) / denom

            total = r + g + b
            total = np.where(total > 0.01, total, 0.01)
            r_norm = r / total
            g_norm = g / total
            b_norm = b / total
            exg = 2 * g_norm - r_norm - b_norm

            ndvi = 0.6 * np.clip(vari, -1, 1) + 0.4 * np.clip(exg * 2, -1, 1)
            ndvi = np.clip(ndvi, -1, 1)

            brightness = (r + g + b) / 3
            valid_mask = (brightness > 0.05) & (brightness < 0.95)
            source_type = "rgb_estimated"

    logger.info(
        f"Raster (effective): {out_w}x{out_h}, {band_count} bands, source={source_type}"
    )

    # Compute gradient for statistics
    grad_x = ndimage.sobel(ndvi, axis=1)
    grad_y = ndimage.sobel(ndvi, axis=0)
    gradient = np.sqrt(grad_x**2 + grad_y**2)

    # Segment fields
    labels = segment_by_gradient(ndvi, valid_mask, config)

    # Convert to polygons
    logger.info("Converting to polygons...")
    polygons = []

    for geom, value in rasterio_shapes(
        labels.astype(np.int32), mask=(labels > 0), transform=transform
    ):
        if value == 0:
            continue
        try:
            poly = shape(geom)
            if poly.is_valid and poly.area > 0:
                polygons.append(poly)
        except:
            continue

    logger.info(f"Initial polygons: {len(polygons)}")

    # OSM road clipping
    if config.use_osm_roads:
        roads = fetch_osm_roads(aoi_bounds)
        if roads:
            polygons = clip_polygons_by_roads(polygons, roads, config.road_buffer_m)

    # Filter and compute statistics
    features = []
    all_zones = []

    # Area thresholds in degrees^2 (approximate)
    min_area_deg = config.min_area_ha * 10000 / (111000 * 87000)
    max_area_deg = config.max_area_ha * 10000 / (111000 * 87000)
    simplify_tol = config.simplify_tolerance_m / 111000

    for i, poly in enumerate(polygons):
        # Filter by area
        if poly.area < min_area_deg or poly.area > max_area_deg:
            continue

        # Simplify
        poly = poly.simplify(simplify_tol, preserve_topology=True)

        if poly.is_empty:
            continue

        # Compute statistics
        stats = compute_field_statistics(poly, ndvi, gradient, transform, config)

        # Compute area in hectares
        area_ha = poly.area * (111000 * 87000) / 10000

        # Compute confidence
        confidence = compute_confidence_v2(stats, source_type)

        # Generate feature ID
        feature_id = hashlib.md5(str(poly.wkt)[:100].encode()).hexdigest()[:12]

        feature = {
            "type": "Feature",
            "id": feature_id,
            "properties": {
                "id": feature_id,
                "field_index": len(features) + 1,
                "area_ha": round(area_ha, 2),
                "source": source_type,
                "confidence": confidence,
                **{
                    k: round(v, 3) if isinstance(v, float) else v
                    for k, v in stats.items()
                },
                "created_at": datetime.utcnow().isoformat() + "Z",
            },
            "geometry": mapping(poly),
        }
        features.append(feature)

        # Generate management zones
        if config.generate_zones and SKLEARN_AVAILABLE:
            zones = generate_management_zones(poly, ndvi, transform, config.num_zones)
            for zone in zones:
                zone_id = f"{feature_id}_z{zone['zone_id']}"
                zone_feature = {
                    "type": "Feature",
                    "id": zone_id,
                    "properties": {
                        "id": zone_id,
                        "parent_field": feature_id,
                        "zone_id": zone["zone_id"],
                        "zone_class": zone["zone_class"],
                        "mean_ndvi": round(zone["mean_ndvi"], 3),
                    },
                    "geometry": mapping(zone["geometry"]),
                }
                all_zones.append(zone_feature)

    logger.info(f"Final field polygons: {len(features)}")
    logger.info(f"Management zones: {len(all_zones)}")

    # Save fields GeoJSON
    fields_geojson = {
        "type": "FeatureCollection",
        "properties": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "source_method": source_type,
            "feature_count": len(features),
            "version": "2.0",
        },
        "features": features,
    }

    fields_path = out_dir / "fields.geojson"
    write_json(fields_geojson, fields_path)
    logger.info(f"Fields saved: {fields_path}")

    # Save zones GeoJSON
    if all_zones:
        zones_geojson = {
            "type": "FeatureCollection",
            "properties": {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "zone_count": len(all_zones),
                "num_classes": config.num_zones,
            },
            "features": all_zones,
        }
        zones_path = out_dir / "zones.geojson"
        write_json(zones_geojson, zones_path)
        logger.info(f"Zones saved: {zones_path}")

    # Save metadata
    metadata = {
        "status": "completed",
        "version": "2.0",
        "source_method": source_type,
        "field_count": len(features),
        "zone_count": len(all_zones),
        "config": {
            "min_area_ha": config.min_area_ha,
            "max_area_ha": config.max_area_ha,
            "use_osm_roads": config.use_osm_roads,
            "generate_zones": config.generate_zones,
            "num_zones": config.num_zones,
        },
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    write_json(metadata, out_dir / "extraction_metadata.json")

    logger.info("=" * 60)
    logger.info(
        f"V2 EXTRACTION COMPLETE: {len(features)} fields, {len(all_zones)} zones"
    )
    logger.info("=" * 60)

    return {
        "status": "completed",
        "fields_path": str(fields_path),
        "zones_path": str(out_dir / "zones.geojson") if all_zones else None,
        "field_count": len(features),
        "zone_count": len(all_zones),
        "source_method": source_type,
    }


# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced field extraction v2")
    parser.add_argument("--aoi", "-a", required=True)
    parser.add_argument("--rasters", "-r", nargs="+", required=True)
    parser.add_argument("--out", "-o", required=True)
    parser.add_argument("--fetch-multiband", action="store_true")
    parser.add_argument("--no-osm", action="store_true")
    parser.add_argument("--no-zones", action="store_true")
    parser.add_argument("--num-zones", type=int, default=3)

    args = parser.parse_args()

    config = ExtractionConfigV2(
        use_osm_roads=not args.no_osm,
        generate_zones=not args.no_zones,
        num_zones=args.num_zones,
    )

    result = extract_field_polygons_v2(
        aoi_geojson=Path(args.aoi),
        raster_paths=[Path(r) for r in args.rasters],
        out_dir=Path(args.out),
        config=config,
        fetch_multiband=args.fetch_multiband,
    )

    print(f"\nExtracted {result['field_count']} fields, {result['zone_count']} zones")
