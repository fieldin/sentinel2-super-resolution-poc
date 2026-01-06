"""
Real-ESRGAN + High Zoom Tiles Generator

Generate tiles at zoom levels 18, 19, 20 from Real-ESRGAN enhanced imagery.

Usage:
    python -m app.esrgan_tiles [--input PATH] [--output-dir PATH] [--min-zoom 18] [--max-zoom 20]
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import shutil

from .utils import setup_logging, find_latest_file, ensure_directory
from .wow_sr import apply_wow_sr
from .tiling import process_raster_to_tiles, get_raster_info, reproject_to_web_mercator

logger = setup_logging("esrgan_tiles")


def run_esrgan_and_tiles(
    input_path: Path,
    output_dir: Path,
    min_zoom: int = 18,
    max_zoom: int = 20,
    enhance_crops: bool = True,
    skip_sr: bool = False,
    sr_output: Path = None,
) -> dict:
    """
    Run Real-ESRGAN super-resolution and generate high-zoom tiles.
    
    Args:
        input_path: Input GeoTIFF (source Sentinel-2 imagery)
        output_dir: Output directory for SR images and tiles
        min_zoom: Minimum zoom level (default 18)
        max_zoom: Maximum zoom level (default 20)
        enhance_crops: Apply crop visibility enhancement
        skip_sr: Skip SR, just generate tiles from existing SR output
        sr_output: Path to existing SR output (required if skip_sr=True)
        
    Returns:
        Dictionary with results and metadata
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "input": str(input_path),
        "min_zoom": min_zoom,
        "max_zoom": max_zoom,
        "steps": []
    }
    
    # Ensure output directories
    sr_dir = output_dir / "sr_esrgan"
    tiles_dir = output_dir / "tiles_esrgan"
    ensure_directory(sr_dir)
    ensure_directory(tiles_dir)
    
    sr_tif = sr_output
    
    # ============================================================
    # Step 1: Real-ESRGAN Super-Resolution
    # ============================================================
    if not skip_sr:
        logger.info("=" * 60)
        logger.info("Step 1/2: Real-ESRGAN x4 Super-Resolution")
        logger.info("=" * 60)
        
        base_name = input_path.stem
        sr_tif = sr_dir / f"{base_name}_esrgan_x4.tif"
        
        try:
            output_path, sr_metadata = apply_wow_sr(
                input_path=input_path,
                output_path=sr_tif,
                enhance_crops=enhance_crops,
            )
            sr_tif = Path(output_path)
            
            results["steps"].append({
                "step": 1,
                "name": "Real-ESRGAN SR",
                "status": "completed",
                "output": str(sr_tif),
                "metadata": sr_metadata,
            })
            logger.info(f"✅ SR complete: {sr_tif}")
            
        except Exception as e:
            logger.error(f"❌ SR failed: {e}")
            results["steps"].append({
                "step": 1,
                "name": "Real-ESRGAN SR",
                "status": "failed",
                "error": str(e),
            })
            return results
    else:
        logger.info("⏭️  Skipping SR (using existing output)")
        results["steps"].append({
            "step": 1,
            "name": "Real-ESRGAN SR",
            "status": "skipped",
            "output": str(sr_tif),
        })
    
    # ============================================================
    # Step 2: Generate High-Zoom Tiles (z18, z19, z20)
    # ============================================================
    logger.info("=" * 60)
    logger.info(f"Step 2/2: Generating Tiles (z{min_zoom}-{max_zoom})")
    logger.info("=" * 60)
    
    try:
        # Get raster info
        info = get_raster_info(sr_tif)
        logger.info(f"SR image: {info.width}x{info.height} pixels, CRS: {info.crs}")
        
        # Reproject to Web Mercator if needed
        if info.crs != "EPSG:3857":
            reprojected_path = sr_tif.parent / f"{sr_tif.stem}_3857.tif"
            working_path = reproject_to_web_mercator(sr_tif, reprojected_path)
            logger.info(f"✅ Reprojected to EPSG:3857: {working_path}")
        else:
            working_path = sr_tif
        
        # Generate tiles
        from .tiling import generate_xyz_tiles, create_tileset_metadata
        
        generate_xyz_tiles(
            working_path,
            tiles_dir,
            min_zoom=min_zoom,
            max_zoom=max_zoom,
            resampling="lanczos"  # High-quality resampling for SR output
        )
        
        # Create metadata
        metadata = create_tileset_metadata(
            tiles_dir,
            info.bounds_4326,
            min_zoom,
            max_zoom,
            tile_template="/tiles_esrgan/{z}/{x}/{y}.png"
        )
        
        # Count generated tiles
        tile_count = sum(1 for _ in tiles_dir.rglob("*.png"))
        
        results["steps"].append({
            "step": 2,
            "name": "Tile Generation",
            "status": "completed",
            "output_dir": str(tiles_dir),
            "tile_count": tile_count,
            "zoom_levels": list(range(min_zoom, max_zoom + 1)),
            "metadata": metadata,
        })
        
        logger.info(f"✅ Generated {tile_count} tiles at z{min_zoom}-{max_zoom}")
        
    except Exception as e:
        logger.error(f"❌ Tile generation failed: {e}")
        import traceback
        traceback.print_exc()
        results["steps"].append({
            "step": 2,
            "name": "Tile Generation",
            "status": "failed",
            "error": str(e),
        })
        return results
    
    # ============================================================
    # Summary
    # ============================================================
    results["status"] = "completed"
    results["sr_output"] = str(sr_tif)
    results["tiles_dir"] = str(tiles_dir)
    results["tile_count"] = tile_count
    
    logger.info("=" * 60)
    logger.info("🎉 Real-ESRGAN + High-Zoom Tiles Complete!")
    logger.info(f"   SR Output: {sr_tif}")
    logger.info(f"   Tiles: {tiles_dir}")
    logger.info(f"   Zoom levels: {min_zoom}-{max_zoom}")
    logger.info(f"   Tile count: {tile_count}")
    logger.info("=" * 60)
    
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Real-ESRGAN enhanced tiles at zoom 18-20",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-find latest source image
  python -m app.esrgan_tiles

  # Specify input file
  python -m app.esrgan_tiles --input /app/data/source/image.tif

  # Custom zoom range
  python -m app.esrgan_tiles --min-zoom 17 --max-zoom 20

  # Use existing SR output (skip SR processing)
  python -m app.esrgan_tiles --skip-sr --sr-output /app/data/wow/sr_image.tif
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        help="Input GeoTIFF path. If not specified, finds latest in source dir."
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="/app/data",
        help="Output directory (default: /app/data)"
    )
    parser.add_argument(
        "--min-zoom",
        type=int,
        default=18,
        help="Minimum zoom level (default: 18)"
    )
    parser.add_argument(
        "--max-zoom",
        type=int,
        default=20,
        help="Maximum zoom level (default: 20)"
    )
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Skip crop enhancement post-processing"
    )
    parser.add_argument(
        "--skip-sr",
        action="store_true",
        help="Skip SR processing, only generate tiles from existing SR output"
    )
    parser.add_argument(
        "--sr-output",
        help="Path to existing SR output (required if --skip-sr is set)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Real-ESRGAN + High-Zoom Tiles Generator")
    logger.info("=" * 60)
    
    # Determine input file
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            sys.exit(1)
    else:
        # Find latest source file
        source_dir = Path(args.output_dir) / "source"
        input_path = find_latest_file(source_dir, "*.tif")
        
        if not input_path:
            logger.error(f"No GeoTIFF files found in {source_dir}")
            logger.error("Run 'python -m app.fetch' first or specify --input")
            sys.exit(1)
    
    logger.info(f"Input: {input_path}")
    
    # Validate skip_sr args
    sr_output = None
    if args.skip_sr:
        if not args.sr_output:
            logger.error("--sr-output is required when using --skip-sr")
            sys.exit(1)
        sr_output = Path(args.sr_output)
        if not sr_output.exists():
            logger.error(f"SR output not found: {sr_output}")
            sys.exit(1)
    
    # Run pipeline
    try:
        result = run_esrgan_and_tiles(
            input_path=input_path,
            output_dir=Path(args.output_dir),
            min_zoom=args.min_zoom,
            max_zoom=args.max_zoom,
            enhance_crops=not args.no_enhance,
            skip_sr=args.skip_sr,
            sr_output=sr_output,
        )
        
        if result.get("status") == "completed":
            logger.info("\n✅ Pipeline completed successfully!")
            logger.info(f"   View tiles at: /tiles_esrgan/{{z}}/{{x}}/{{y}}.png")
            sys.exit(0)
        else:
            logger.error("\n❌ Pipeline failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()








