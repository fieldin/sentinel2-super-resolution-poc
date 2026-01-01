"""
GDAL helpers for reprojection and tile generation.
"""
import subprocess
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from .utils import setup_logging, ensure_directory, write_json

logger = setup_logging("tiling")


@dataclass
class RasterInfo:
    """Information about a raster file."""
    path: Path
    crs: str
    bounds: list  # [west, south, east, north] in native CRS
    bounds_4326: list  # [west, south, east, north] in EPSG:4326
    width: int
    height: int
    bands: int
    dtype: str


def get_raster_info(raster_path: Path) -> RasterInfo:
    """
    Get information about a raster file using gdalinfo.
    
    Args:
        raster_path: Path to raster file
        
    Returns:
        RasterInfo object with raster metadata
    """
    logger.info(f"Getting raster info: {raster_path}")
    
    # Run gdalinfo -json
    result = subprocess.run(
        ["gdalinfo", "-json", str(raster_path)],
        capture_output=True,
        text=True,
        check=True
    )
    
    info = json.loads(result.stdout)
    
    # Extract CRS
    crs = "EPSG:4326"  # default
    if "coordinateSystem" in info and "wkt" in info["coordinateSystem"]:
        wkt = info["coordinateSystem"]["wkt"]
        # Try to extract EPSG code from WKT
        if "AUTHORITY" in wkt:
            import re
            match = re.search(r'AUTHORITY\["EPSG","(\d+)"\]', wkt)
            if match:
                crs = f"EPSG:{match.group(1)}"
    
    # Get bounds
    corner_coords = info.get("cornerCoordinates", {})
    ul = corner_coords.get("upperLeft", [0, 0])
    lr = corner_coords.get("lowerRight", [0, 0])
    bounds = [ul[0], lr[1], lr[0], ul[1]]  # west, south, east, north
    
    # Get bounds in EPSG:4326 if different CRS
    wgs84_extent = info.get("wgs84Extent", {})
    if wgs84_extent and "coordinates" in wgs84_extent:
        coords = wgs84_extent["coordinates"][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        bounds_4326 = [min(lons), min(lats), max(lons), max(lats)]
    else:
        bounds_4326 = bounds  # Assume already in 4326
    
    # Get size
    size = info.get("size", [0, 0])
    
    # Get bands
    bands = len(info.get("bands", []))
    
    # Get data type
    dtype = "Byte"
    if info.get("bands"):
        dtype = info["bands"][0].get("type", "Byte")
    
    return RasterInfo(
        path=raster_path,
        crs=crs,
        bounds=bounds,
        bounds_4326=bounds_4326,
        width=size[0],
        height=size[1],
        bands=bands,
        dtype=dtype
    )


def reproject_to_web_mercator(
    input_path: Path,
    output_path: Path,
    resample_method: str = "bilinear"
) -> Path:
    """
    Reproject raster to EPSG:3857 (Web Mercator) for tile generation.
    
    Args:
        input_path: Input raster path
        output_path: Output raster path
        resample_method: GDAL resampling method
        
    Returns:
        Path to reprojected raster
    """
    logger.info(f"Reprojecting to EPSG:3857: {input_path}")
    
    ensure_directory(output_path.parent)
    
    cmd = [
        "gdalwarp",
        "-t_srs", "EPSG:3857",
        "-r", resample_method,
        "-co", "COMPRESS=LZW",
        "-co", "TILED=YES",
        "-overwrite",
        str(input_path),
        str(output_path)
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, capture_output=True)
    
    logger.info(f"Reprojection complete: {output_path}")
    return output_path


def generate_xyz_tiles(
    input_path: Path,
    output_dir: Path,
    min_zoom: int = 10,
    max_zoom: int = 16,
    tile_size: int = 256,
    resampling: str = "average"
) -> Path:
    """
    Generate XYZ tiles using gdal2tiles.py.
    
    Args:
        input_path: Input raster path (should be in EPSG:3857)
        output_dir: Output directory for tiles
        min_zoom: Minimum zoom level
        max_zoom: Maximum zoom level
        tile_size: Tile size in pixels
        resampling: Resampling method
        
    Returns:
        Path to tiles directory
    """
    logger.info(f"Generating XYZ tiles: zoom {min_zoom}-{max_zoom}")
    
    ensure_directory(output_dir)
    
    # gdal2tiles.py command
    cmd = [
        "gdal2tiles.py",
        "--zoom", f"{min_zoom}-{max_zoom}",
        "--tilesize", str(tile_size),
        "--resampling", resampling,
        "--xyz",  # Use XYZ tile naming (TMS by default uses inverted Y)
        "--processes", "4",  # Parallel processing
        "--webviewer", "none",  # Don't generate HTML viewer
        str(input_path),
        str(output_dir)
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"gdal2tiles.py failed: {e.stderr}")
        raise
    
    logger.info(f"Tile generation complete: {output_dir}")
    return output_dir


def create_tileset_metadata(
    tiles_dir: Path,
    bounds_4326: list,
    min_zoom: int,
    max_zoom: int,
    tile_template: str = "/tiles/{z}/{x}/{y}.png"
) -> dict:
    """
    Create tileset metadata JSON.
    
    Args:
        tiles_dir: Directory containing tiles
        bounds_4326: Bounds in EPSG:4326 [west, south, east, north]
        min_zoom: Minimum zoom level
        max_zoom: Maximum zoom level
        tile_template: URL template for tiles
        
    Returns:
        Tileset metadata dictionary
    """
    metadata = {
        "bounds": bounds_4326,
        "minzoom": min_zoom,
        "maxzoom": max_zoom,
        "tileTemplate": tile_template,
        "attribution": "Sentinel-2 SR via UP42",
        "format": "png",
        "tileSize": 256
    }
    
    metadata_path = tiles_dir / "tileset.json"
    write_json(metadata, metadata_path)
    
    logger.info(f"Tileset metadata saved: {metadata_path}")
    return metadata


def process_raster_to_tiles(
    input_path: Path,
    tiles_dir: Path,
    min_zoom: int = 10,
    max_zoom: int = 16
) -> dict:
    """
    Complete pipeline: check CRS, reproject if needed, generate tiles.
    
    Args:
        input_path: Input GeoTIFF path
        tiles_dir: Output directory for tiles
        min_zoom: Minimum zoom level
        max_zoom: Maximum zoom level
        
    Returns:
        Tileset metadata dictionary
    """
    logger.info(f"Processing raster to tiles: {input_path}")
    
    # Get raster info
    info = get_raster_info(input_path)
    logger.info(f"Raster CRS: {info.crs}")
    logger.info(f"Raster bounds (4326): {info.bounds_4326}")
    
    # Reproject to Web Mercator if needed
    if info.crs != "EPSG:3857":
        reprojected_path = input_path.parent / f"{input_path.stem}_3857.tif"
        working_path = reproject_to_web_mercator(input_path, reprojected_path)
    else:
        working_path = input_path
    
    # Generate tiles
    generate_xyz_tiles(
        working_path,
        tiles_dir,
        min_zoom=min_zoom,
        max_zoom=max_zoom
    )
    
    # Create and return metadata
    metadata = create_tileset_metadata(
        tiles_dir,
        info.bounds_4326,
        min_zoom,
        max_zoom
    )
    
    return metadata

