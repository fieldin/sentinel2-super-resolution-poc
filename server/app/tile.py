"""
CLI module for generating XYZ tiles from downloaded imagery.

Usage:
    python -m app.tile
"""
import sys
from pathlib import Path

from .settings import get_settings
from .tiling import process_raster_to_tiles
from .utils import setup_logging, find_latest_file, ensure_directory

logger = setup_logging("tile")


def main():
    """
    Main tiling pipeline:
    1. Find latest GeoTIFF in source directory
    2. Reproject to Web Mercator if needed
    3. Generate XYZ tiles
    4. Create tileset metadata
    """
    logger.info("=" * 60)
    logger.info("Tile Generation Pipeline")
    logger.info("=" * 60)
    
    try:
        settings = get_settings()
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        sys.exit(1)
    
    # Find latest GeoTIFF
    source_dir = Path(settings.data_dir) / "source"
    latest_tif = find_latest_file(source_dir, "*.tif")
    
    if not latest_tif:
        logger.error(f"No GeoTIFF files found in {source_dir}")
        logger.error("Run 'python -m app.fetch' first to download imagery.")
        sys.exit(1)
    
    logger.info(f"Processing: {latest_tif}")
    
    # Setup tiles directory
    tiles_dir = Path(settings.data_dir) / "tiles"
    ensure_directory(tiles_dir)
    
    try:
        # Process raster to tiles
        metadata = process_raster_to_tiles(
            input_path=latest_tif,
            tiles_dir=tiles_dir,
            min_zoom=settings.tile_min_zoom,
            max_zoom=settings.tile_max_zoom
        )
        
        logger.info("=" * 60)
        logger.info("Tiling Complete!")
        logger.info(f"  Tiles directory: {tiles_dir}")
        logger.info(f"  Zoom range: {metadata['minzoom']}-{metadata['maxzoom']}")
        logger.info(f"  Bounds: {metadata['bounds']}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Tiling failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

