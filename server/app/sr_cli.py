#!/usr/bin/env python3
"""
CLI for running Super-Resolution on Sentinel-2 imagery.

Usage:
    python -m app.sr_cli [--input FILE] [--scale 2|3|4] [--output DIR]
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Apply AI Super-Resolution to Sentinel-2 imagery"
    )
    parser.add_argument(
        "-i", "--input",
        help="Input GeoTIFF file (default: latest in data/source)",
    )
    parser.add_argument(
        "-o", "--output",
        default="./data/sr",
        help="Output directory (default: ./data/sr)",
    )
    parser.add_argument(
        "-s", "--scale",
        type=int,
        choices=[2, 3, 4],
        default=4,
        help="SR scale factor: 2 (5m), 3 (3.3m), or 4 (2.5m effective resolution)",
    )
    parser.add_argument(
        "-m", "--model",
        choices=["edsr", "espcn", "lapsrn"],
        default="edsr",
        help="SR model type (default: edsr)",
    )
    parser.add_argument(
        "--tile",
        action="store_true",
        help="Also generate XYZ tiles from SR output",
    )
    parser.add_argument(
        "--tile-min-zoom",
        type=int,
        default=10,
        help="Minimum tile zoom level",
    )
    parser.add_argument(
        "--tile-max-zoom",
        type=int,
        default=18,
        help="Maximum tile zoom level",
    )

    args = parser.parse_args()

    # Find input file
    if args.input:
        input_file = Path(args.input)
    else:
        # Find latest GeoTIFF
        source_dir = Path("./data/source")
        if not source_dir.exists():
            print("❌ No data/source directory found. Run fetch first.")
            sys.exit(1)
        
        tif_files = sorted(source_dir.glob("*.tif"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not tif_files:
            print("❌ No GeoTIFF files found in data/source")
            sys.exit(1)
        
        input_file = tif_files[0]
        print(f"📂 Using latest file: {input_file}")

    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)

    output_dir = Path(args.output)
    
    print("=" * 60)
    print("🔬 Sentinel-2 Super-Resolution")
    print("=" * 60)
    print(f"   Input: {input_file}")
    print(f"   Model: {args.model.upper()} x{args.scale}")
    print(f"   Output: {output_dir}")
    print(f"   Original resolution: 10m")
    print(f"   Effective resolution: {10 / args.scale:.1f}m")
    print()

    # Run SR
    from .super_resolution import process_sentinel2_sr
    
    result = process_sentinel2_sr(
        input_tif=input_file,
        output_dir=output_dir,
        scale=args.scale,
        model_type=args.model,
    )

    # Generate tiles if requested
    if args.tile:
        sr_tif = result["outputs"].get("sr_tif")
        if sr_tif and Path(sr_tif).exists():
            print("\n" + "=" * 60)
            print("🗺️  Generating XYZ Tiles")
            print("=" * 60)
            
            from .tiling import generate_tiles
            
            tiles_dir = Path("./data/tiles_sr")
            generate_tiles(
                input_file=Path(sr_tif),
                output_dir=tiles_dir,
                min_zoom=args.tile_min_zoom,
                max_zoom=args.tile_max_zoom,
            )
            result["tiles_dir"] = str(tiles_dir)
            print(f"\n   ✅ Tiles saved to: {tiles_dir}")

    print("\n" + "=" * 60)
    print("✅ COMPLETE!")
    print("=" * 60)
    print(f"\nOutputs:")
    for name, path in result["outputs"].items():
        if path:
            print(f"   {name}: {path}")
    
    meta = result.get("sr_metadata", {})
    print(f"\nEffective resolution: {meta.get('effective_resolution_m', '?')}m")
    print(f"GPU used: {'Yes' if meta.get('gpu_used') else 'No (CPU)'}")


if __name__ == "__main__":
    main()

