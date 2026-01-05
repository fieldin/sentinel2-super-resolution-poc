#!/usr/bin/env python3
"""
CLI command to generate field boundary vectors from satellite imagery.

Usage:
    python -m app.generate_vectors --aoi config/aoi.geojson --out data/vectors
    python -m app.generate_vectors --aoi config/aoi.geojson --rasters data/wow/latest/sr.tif --out data/vectors

This is a convenience wrapper around the vector_extraction module that:
- Auto-discovers raster files if not specified
- Sets sensible defaults for POC usage
- Provides clear progress output
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

from .vector_extraction import extract_field_polygons, ExtractionConfig
from .utils import setup_logging, find_latest_file

logger = setup_logging("generate_vectors")


def find_best_raster(data_dir: Path) -> Path:
    """
    Auto-discover the best raster file to use for vector extraction.

    Priority:
    1. WOW SR output (best quality)
    2. Standard SR output
    3. Original source imagery

    Args:
        data_dir: Base data directory

    Returns:
        Path to the best available raster
    """
    # Priority order for raster sources
    search_paths = [
        (data_dir / "wow", "WOW SR"),
        (data_dir / "sr", "Standard SR"),
        (data_dir / "source", "Original source"),
    ]

    for search_dir, source_name in search_paths:
        if search_dir.exists():
            # Find most recent .tif file (recursively)
            tif_files = list(search_dir.rglob("*.tif"))
            if tif_files:
                latest = max(tif_files, key=lambda f: f.stat().st_mtime)
                logger.info(f"Found {source_name}: {latest}")
                return latest

    return None


def main():
    """Main entry point for vector generation CLI."""
    parser = argparse.ArgumentParser(
        description="Generate field boundary vectors from satellite imagery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover raster, use default AOI
  python -m app.generate_vectors

  # Specify AOI and output directory
  python -m app.generate_vectors --aoi config/aoi.geojson --out data/vectors

  # Use specific raster file
  python -m app.generate_vectors --rasters data/wow/latest/sr.tif --out data/vectors

  # Custom extraction parameters
  python -m app.generate_vectors --ndvi-threshold 0.25 --min-area 0.5 --simplify 3.0
        """,
    )

    parser.add_argument(
        "--aoi",
        "-a",
        type=str,
        default="config/aoi.geojson",
        help="Path to AOI GeoJSON file (default: config/aoi.geojson)",
    )
    parser.add_argument(
        "--rasters",
        "-r",
        type=str,
        nargs="*",
        help="Path(s) to input raster file(s). If not specified, auto-discovers from data/",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=str,
        default="data/vectors",
        help="Output directory for vectors (default: data/vectors)",
    )
    parser.add_argument(
        "--ndvi-threshold",
        type=float,
        default=0.3,
        help="NDVI threshold for vegetation detection (default: 0.3)",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.1,
        help="Minimum field area in hectares (default: 0.1)",
    )
    parser.add_argument(
        "--max-area",
        type=float,
        default=500.0,
        help="Maximum field area in hectares (default: 500)",
    )
    parser.add_argument(
        "--simplify",
        type=float,
        default=5.0,
        help="Simplification tolerance in meters (default: 5)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    print()
    print("=" * 60)
    print("🗺️  FIELD BOUNDARY VECTOR EXTRACTION")
    print("=" * 60)
    print()

    # Resolve AOI path
    aoi_path = Path(args.aoi)
    if not aoi_path.exists():
        # Try with /app prefix (Docker environment)
        aoi_path = Path("/app") / args.aoi

    if not aoi_path.exists():
        logger.error(f"AOI file not found: {args.aoi}")
        print(f"\n❌ Error: AOI file not found: {args.aoi}")
        print("   Please provide a valid AOI GeoJSON file.")
        sys.exit(1)

    print(f"📍 AOI: {aoi_path}")

    # Resolve raster paths
    if args.rasters:
        raster_paths = [Path(r) for r in args.rasters]
    else:
        # Auto-discover raster
        data_dir = Path("data")
        if not data_dir.exists():
            data_dir = Path("/app/data")

        best_raster = find_best_raster(data_dir)
        if best_raster is None:
            logger.error("No raster files found. Run the pipeline first.")
            print("\n❌ Error: No raster files found.")
            print("   Run the pipeline first: make pipeline")
            sys.exit(1)

        raster_paths = [best_raster]

    # Validate raster paths
    valid_rasters = [p for p in raster_paths if p.exists()]
    if not valid_rasters:
        logger.error(f"No valid raster files found: {raster_paths}")
        print(f"\n❌ Error: Raster file(s) not found: {raster_paths}")
        sys.exit(1)

    print(f"🛰️  Raster: {valid_rasters[0]}")

    # Resolve output directory
    out_dir = Path(args.out)
    print(f"📁 Output: {out_dir}")

    # Create extraction config
    config = ExtractionConfig(
        ndvi_threshold=args.ndvi_threshold,
        min_area_ha=args.min_area,
        max_area_ha=args.max_area,
        simplify_tolerance_m=args.simplify,
    )

    print()
    print(f"⚙️  Configuration:")
    print(f"   NDVI threshold: {config.ndvi_threshold}")
    print(f"   Min area: {config.min_area_ha} ha")
    print(f"   Max area: {config.max_area_ha} ha")
    print(f"   Simplify: {config.simplify_tolerance_m}m")
    print()

    # Run extraction
    start_time = datetime.now()

    try:
        result = extract_field_polygons(
            aoi_geojson=aoi_path,
            raster_paths=valid_rasters,
            out_dir=out_dir,
            config=config,
        )
    except ImportError as e:
        print(f"\n❌ Error: Missing dependency - {e}")
        print("   Install shapely: pip install shapely")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        print(f"\n❌ Error: Extraction failed - {e}")
        sys.exit(1)

    elapsed = (datetime.now() - start_time).total_seconds()

    print()
    print("=" * 60)
    print(f"✅ EXTRACTION COMPLETE")
    print("=" * 60)
    print()
    print(f"📊 Results:")
    print(f"   Fields extracted: {result['feature_count']}")
    print(f"   Method: {result['source_method']}")
    print(f"   Time: {elapsed:.1f}s")
    print()
    print(f"📄 Output: {result['output_path']}")
    print()
    print("🗺️  View on map: http://localhost:8080")
    print("   Toggle 'Field Boundaries' in the control panel")
    print()


if __name__ == "__main__":
    main()
