"""
WOW Super-Resolution: Real-ESRGAN x4 with Enhanced Post-Processing

This provides SUPERIOR results at high zoom levels (z18) by:
1. Real-ESRGAN x4: High-quality GAN-based upscaling
2. Enhanced post-processing: CLAHE + unsharp mask for crop visibility
3. Vegetation enhancement for agricultural imagery

Total: x4 upscaling (10m → 2.5m effective resolution) optimized for z18.

Note: Real-ESRGAN x4 provides excellent quality. The publicly available x2
model uses 12 input channels (for USM preprocessing) which is incompatible
with standard RGB images.
"""

import numpy as np
import cv2
import rasterio
from rasterio.transform import Affine
from pathlib import Path
from typing import Tuple
import json
from datetime import datetime

from app.cnn_super_resolution import RealESRGAN


def apply_wow_sr(
    input_path: Path,
    output_path: Path,
    enhance_crops: bool = True,
) -> Tuple[Path, dict]:
    """
    Apply WOW Super-Resolution: Real-ESRGAN x4 with enhanced post-processing.

    Pipeline:
    1. Real-ESRGAN x4 - High-quality GAN upscaling
    2. Post-processing - CLAHE + unsharp mask + vegetation boost

    Total scale: x4 (10m → 2.5m effective resolution)
    """
    print(f"\n🌟 WOW Super-Resolution (Real-ESRGAN x4 + Enhanced)")
    print(f"   Input: {input_path}")
    print(f"   Strategy: High-quality GAN upscaling with crop optimization")

    input_path = Path(input_path)
    transform = None
    crs = None

    # Read input
    if input_path.suffix.lower() in [".tif", ".tiff"]:
        with rasterio.open(input_path) as src:
            if src.count >= 3:
                img = np.stack([src.read(i) for i in [1, 2, 3]], axis=-1)
            else:
                img = src.read(1)
                img = np.stack([img, img, img], axis=-1)

            if img.dtype != np.uint8:
                if img.max() > 255:
                    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(
                        np.uint8
                    )
                else:
                    img = img.astype(np.uint8)

            transform = src.transform
            crs = src.crs
    else:
        img = cv2.imread(str(input_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    original_shape = img.shape[:2]
    print(f"   Input size: {img.shape[1]}x{img.shape[0]} pixels")

    # Convert to BGR for models
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    pipeline_stages = []

    # ============================================================
    # Stage 1: Real-ESRGAN x4 (High-Quality GAN Upscaling)
    # ============================================================
    print(f"\n   🎨 Stage 1/2: Real-ESRGAN x4 (GAN upscaling)...")
    esrgan = RealESRGAN(scale=4, tile_size=256)
    sr_bgr = esrgan.enhance(img_bgr)
    print(f"      → {sr_bgr.shape[1]}x{sr_bgr.shape[0]} pixels")
    del esrgan
    pipeline_stages.append(
        {"model": "RealESRGAN_x4", "purpose": "High-quality GAN upscaling"}
    )

    # Convert to RGB
    output_rgb = cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2RGB)

    # ============================================================
    # Stage 2: Enhanced post-processing for crops
    # ============================================================
    if enhance_crops:
        print(f"\n   🌾 Stage 2/2: Crop visibility enhancement...")
        output_rgb = _enhance_for_crops(output_rgb)
        pipeline_stages.append(
            {"post_processing": "Enhanced", "purpose": "Crop visibility"}
        )

    final_shape = output_rgb.shape[:2]
    actual_scale = 4

    print(f"\n   📊 Final size: {output_rgb.shape[1]}x{output_rgb.shape[0]} pixels")
    print(f"   📊 Total scale: x{actual_scale}")
    print(f"   📊 Pipeline: Real-ESRGAN x4 + Enhanced")

    # Save output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if transform is not None:
        # Save as GeoTIFF
        new_transform = Affine(
            transform.a / actual_scale,
            transform.b,
            transform.c,
            transform.d,
            transform.e / actual_scale,
            transform.f,
        )

        output_tif = output_path.with_suffix(".tif")
        with rasterio.open(
            output_tif,
            "w",
            driver="GTiff",
            height=output_rgb.shape[0],
            width=output_rgb.shape[1],
            count=3,
            dtype="uint8",
            crs=crs,
            transform=new_transform,
            compress="lzw",
        ) as dst:
            for i in range(3):
                dst.write(output_rgb[:, :, i], i + 1)

        print(f"   ✅ Saved GeoTIFF: {output_tif}")
        final_output = output_tif
    else:
        output_png = output_path.with_suffix(".png")
        cv2.imwrite(str(output_png), cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR))
        print(f"   ✅ Saved PNG: {output_png}")
        final_output = output_png

    # Also save PNG
    output_png = output_path.with_suffix(".png")
    cv2.imwrite(str(output_png), cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR))
    print(f"   ✅ Saved PNG: {output_png}")

    metadata = {
        "input_file": str(input_path),
        "output_file": str(final_output),
        "scale": actual_scale,
        "pipeline": "Real-ESRGAN x4 + Enhanced",
        "stages": pipeline_stages,
        "enhancements": (
            ["CLAHE local contrast", "Unsharp mask", "Vegetation boost"]
            if enhance_crops
            else []
        ),
        "original_size": list(original_shape),
        "output_size": list(final_shape),
        "original_resolution_m": 10.0,
        "effective_resolution_m": 10.0 / actual_scale,
        "optimized_for": "z18_crop_visibility",
    }

    return final_output, metadata


def _enhance_for_crops(img: np.ndarray) -> np.ndarray:
    """Apply enhancements optimized for crop/agricultural imagery visibility."""
    # Step 1: CLAHE for local contrast (great for crop rows)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Step 2: Unsharp mask for edge sharpening (crop row boundaries)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
    sharpened = cv2.addWeighted(enhanced, 1.4, blurred, -0.4, 0)

    # Step 3: Vegetation enhancement (boost green channel for crops)
    hsv = cv2.cvtColor(sharpened, cv2.COLOR_RGB2HSV).astype(np.float32)
    # Detect green areas (hue ~35-85 in HSV)
    green_mask = ((hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 85)).astype(np.float32)
    # Boost saturation for green areas
    hsv[:, :, 1] = np.where(
        green_mask > 0, np.clip(hsv[:, :, 1] * 1.2, 0, 255), hsv[:, :, 1]
    )
    final = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return np.clip(final, 0, 255).astype(np.uint8)


def process_wow_sr(
    input_tif: Path,
    output_dir: Path,
    enhance_crops: bool = True,
) -> dict:
    """
    Process Sentinel-2 imagery with WOW super-resolution.

    Args:
        input_tif: Path to Sentinel-2 GeoTIFF
        output_dir: Output directory
        enhance_crops: Apply crop visibility enhancement

    Returns:
        Dictionary with output paths and metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(input_tif).stem
    wow_tif = output_dir / f"{base_name}_wow_sr.tif"

    output_path, sr_metadata = apply_wow_sr(
        input_path=input_tif,
        output_path=wow_tif,
        enhance_crops=enhance_crops,
    )

    result = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "input": str(input_tif),
        "outputs": {
            "sr_tif": str(wow_tif) if wow_tif.exists() else None,
            "sr_png": (
                str(wow_tif.with_suffix(".png"))
                if wow_tif.with_suffix(".png").exists()
                else None
            ),
        },
        "sr_metadata": sr_metadata,
    }

    meta_file = output_dir / f"{base_name}_wow_sr_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n🌟 WOW Super-Resolution complete!")
    print(f"   Pipeline: {sr_metadata['pipeline']}")
    print(f"   Effective resolution: {sr_metadata['effective_resolution_m']}m")
    print(f"   Optimized for: z18 crop visibility")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="WOW Super-Resolution (SwinIR → Real-ESRGAN)"
    )
    parser.add_argument("input", help="Input GeoTIFF file")
    parser.add_argument(
        "-o", "--output", help="Output directory", default="./wow_sr_output"
    )
    parser.add_argument(
        "--no-enhance", action="store_true", help="Skip crop enhancement"
    )

    args = parser.parse_args()

    result = process_wow_sr(
        input_tif=Path(args.input),
        output_dir=Path(args.output),
        enhance_crops=not args.no_enhance,
    )

    print(f"\nResults: {result['outputs']}")
