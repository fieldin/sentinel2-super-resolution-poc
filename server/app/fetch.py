"""
CLI module for fetching Sentinel-2 imagery from UP42.

Usage:
    python -m app.fetch
"""

import sys
from pathlib import Path

from .settings import get_settings
from .up42_client import UP42Client, PublicSentinel2Client
from .utils import setup_logging, read_json, ensure_directory

logger = setup_logging("fetch")


def main():
    """
    Main fetch pipeline:
    1. Read AOI from config
    2. Search UP42 catalog for best scene
    3. Download and clip to AOI
    4. Save metadata
    """
    logger.info("=" * 60)
    logger.info("UP42 Sentinel-2 Fetch Pipeline")
    logger.info("=" * 60)

    try:
        settings = get_settings()
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        logger.error("Ensure all required environment variables are set.")
        sys.exit(1)

    # Load AOI
    aoi_path = Path(settings.aoi_path)
    if not aoi_path.exists():
        logger.error(f"AOI file not found: {aoi_path}")
        sys.exit(1)

    logger.info(f"Loading AOI from: {aoi_path}")
    aoi_data = read_json(aoi_path)

    # Extract geometry - handle both Feature and raw Geometry
    if aoi_data.get("type") == "FeatureCollection":
        aoi_geometry = aoi_data["features"][0]["geometry"]
    elif aoi_data.get("type") == "Feature":
        aoi_geometry = aoi_data["geometry"]
    else:
        aoi_geometry = aoi_data

    logger.info(f"AOI type: {aoi_geometry.get('type')}")

    # Setup output directory
    source_dir = Path(settings.data_dir) / "source"
    ensure_directory(source_dir)

    # Initialize client
    # UP42's catalog API is currently not accessible, use public Sentinel-2 from AWS
    # Set USE_UP42=true in env to force UP42 client (requires working API)
    import os

    force_up42 = os.environ.get("USE_UP42", "").lower() == "true"

    if force_up42:
        logger.info("Using UP42 client (USE_UP42=true)")
        client = UP42Client(settings)
    else:
        logger.info("=" * 60)
        logger.info("Using AWS Earth Search for real Sentinel-2 L2A data")
        logger.info("(Set USE_UP42=true to use UP42 API instead)")
        logger.info("=" * 60)
        client = PublicSentinel2Client(settings)

    try:
        # Fetch best scene
        output_path, metadata = client.fetch_best_scene(aoi_geometry, source_dir)

        logger.info("=" * 60)
        logger.info("Fetch Complete!")
        logger.info(f"  Scene ID: {metadata.get('scene_id')}")
        logger.info(f"  Acquisition: {metadata.get('acquisition_date')}")
        logger.info(f"  Cloud Cover: {metadata.get('cloud_cover_pct')}%")
        logger.info(f"  File: {output_path}")
        logger.info(f"  Size: {metadata.get('file_size_mb', 0):.2f} MB")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Fetch failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
