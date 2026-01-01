#!/usr/bin/env python3
"""
Unified Super-Resolution CLI for Sentinel-2 Imagery.

Supports:
- Real-ESRGAN (raw AI SR)
- Farm SR (Real-ESRGAN + crop row enhancements)
"""

import argparse
import glob
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Super-Resolution for Sentinel-2 Imagery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Farm SR (recommended for agriculture)
  python -m app.sr_cli --mode farm --scale 4

  # Raw Real-ESRGAN
  python -m app.sr_cli --mode realesrgan --scale 4

  # Specify input/output
  python -m app.sr_cli --input /path/to/image.tif --output /path/to/output
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["farm", "realesrgan", "edsr"],
        default="farm",
        help="SR mode: 'farm' (crop-optimized), 'realesrgan' (AI SR), or 'edsr' (fast/light)",
    )
    parser.add_argument(
        "--input",
        "-i",
        help="Input GeoTIFF (default: latest in /app/data/source/)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="/app/data/sr",
        help="Output directory",
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=int,
        choices=[2, 4],
        default=4,
        help="Upscaling factor (2 or 4)",
    )
    parser.add_argument(
        "--tile",
        action="store_true",
        help="Also generate XYZ tiles after SR",
    )
    parser.add_argument(
        "--tile-output",
        default="/app/data/tiles_sr",
        help="Tile output directory",
    )
    parser.add_argument(
        "--tile-max-zoom",
        type=int,
        default=20,
        help="Maximum tile zoom level",
    )

    args = parser.parse_args()

    # Find input file
    if args.input:
        input_path = Path(args.input)
    else:
        source_files = sorted(glob.glob("/app/data/source/*_sentinel2_sr.tif"))
        if not source_files:
            print("❌ No source file found in /app/data/source/")
            print("   Run 'make fetch' first to download Sentinel-2 imagery")
            return 1
        input_path = Path(source_files[-1])

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🔬 Sentinel-2 Super-Resolution")
    print("=" * 60)
    print(f"   Input: {input_path}")
    print(f"   Mode: {args.mode.upper()}")
    print(f"   Scale: x{args.scale}")
    print(f"   Output: {output_dir}")
    print()

    # Run SR based on mode
    if args.mode == "farm":
        from app.farm_sr import process_farm_sr

        result = process_farm_sr(
            input_tif=input_path,
            output_dir=output_dir,
            scale=args.scale,
        )
        sr_tif = result["outputs"]["sr_tif"]

    elif args.mode == "realesrgan":
        from app.cnn_super_resolution import apply_cnn_sr

        base_name = input_path.stem
        output_path = output_dir / f"{base_name}_realesrgan_x{args.scale}"

        sr_tif, metadata = apply_cnn_sr(
            input_path=input_path,
            output_path=output_path,
            scale=args.scale,
        )
        sr_tif = str(sr_tif)

    else:  # edsr (fast/light)
        from app.super_resolution import process_sentinel2_sr

        result = process_sentinel2_sr(
            input_tif=input_path,
            output_dir=output_dir,
            scale=args.scale,
            model_type="edsr",
        )
        sr_tif = result["outputs"]["sr_tif"]

    print()
    print("=" * 60)
    print("✅ Super-Resolution Complete!")
    print("=" * 60)
    print(f"   Output: {sr_tif}")

    # Generate tiles if requested
    if args.tile and sr_tif:
        print()
        print("🗺️ Generating XYZ tiles...")
        from app.tiling import process_raster_to_tiles

        process_raster_to_tiles(
            Path(sr_tif),
            Path(args.tile_output),
            min_zoom=10,
            max_zoom=args.tile_max_zoom,
        )
        print(f"   Tiles: {args.tile_output}")

    return 0


if __name__ == "__main__":
    exit(main())
