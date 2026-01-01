"""
Farm-optimized Super-Resolution with Edge Enhancement for Crop Row Visibility.
Uses EDSR + agricultural-specific post-processing for better crop row detection.
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


def enhance_crop_rows(img: np.ndarray, strength: float = 1.5) -> np.ndarray:
    """
    Enhance crop row visibility using directional filtering.
    Detects and enhances parallel line structures typical in farmland.
    """
    # Convert to grayscale for edge detection
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    # Multi-directional edge detection for crop rows
    # Crop rows typically run in specific directions
    edges_combined = np.zeros_like(gray, dtype=np.float32)

    # Sobel filters for different orientations
    kernels = [
        # Horizontal rows
        np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
        # Vertical rows
        np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
        # Diagonal 45°
        np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype=np.float32),
        # Diagonal 135°
        np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=np.float32),
    ]

    for kernel in kernels:
        edges = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        edges_combined += np.abs(edges)

    # Normalize edges
    edges_combined = edges_combined / len(kernels)
    edges_combined = np.clip(edges_combined, 0, 255).astype(np.uint8)

    # Enhance edges with adaptive thresholding for line detection
    edges_enhanced = cv2.adaptiveThreshold(
        edges_combined, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return edges_enhanced


def apply_unsharp_mask(
    img: np.ndarray, strength: float = 1.5, radius: float = 1.0
) -> np.ndarray:
    """Apply unsharp masking for edge sharpening."""
    # Gaussian blur
    blurred = cv2.GaussianBlur(img, (0, 0), radius)

    # Unsharp mask: original + strength * (original - blurred)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_local_contrast(
    img: np.ndarray, clip_limit: float = 3.0, grid_size: int = 8
) -> np.ndarray:
    """Apply CLAHE for local contrast enhancement - great for vegetation patterns."""
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    # Convert back to RGB
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return enhanced


def enhance_vegetation(img: np.ndarray) -> np.ndarray:
    """Enhance green vegetation visibility - useful for crop detection."""
    # Increase saturation in green areas
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)

    # Detect green areas (hue 35-85)
    green_mask = ((hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 85)).astype(np.float32)

    # Boost saturation for green areas
    saturation_boost = 1.3
    hsv[:, :, 1] = np.where(
        green_mask > 0, np.clip(hsv[:, :, 1] * saturation_boost, 0, 255), hsv[:, :, 1]
    )

    # Convert back
    enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return enhanced


def apply_farm_sr(
    input_path: Path,
    output_path: Path,
    scale: int = 4,
) -> Tuple[Path, dict]:
    """
    Apply Farm-optimized Super-Resolution with edge enhancement.

    Pipeline:
    1. EDSR x4 upscaling
    2. Local contrast enhancement (CLAHE)
    3. Unsharp masking for crop rows
    4. Vegetation enhancement
    """
    print(f"\n🌾 Farm Super-Resolution x{scale}")
    print(f"   Input: {input_path}")
    print(f"   Optimized for: Crop row visibility")

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

    # Step 1: Real-ESRGAN Super-Resolution
    print(f"   Step 1/4: Real-ESRGAN x{scale} upscaling...")
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    esrgan = RealESRGAN(scale=scale, tile_size=256)
    sr_bgr = esrgan.enhance(img_bgr)
    sr_rgb = cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2RGB)
    actual_scale = scale
    print(f"            Output: {sr_rgb.shape[1]}x{sr_rgb.shape[0]} pixels")

    # Step 2: Local Contrast Enhancement (CLAHE)
    print(f"   Step 2/4: Local contrast enhancement (CLAHE)...")
    enhanced = enhance_local_contrast(sr_rgb, clip_limit=2.5, grid_size=8)

    # Step 3: Unsharp Masking for crop rows
    print(f"   Step 3/4: Edge sharpening for crop rows...")
    sharpened = apply_unsharp_mask(enhanced, strength=1.2, radius=1.5)

    # Step 4: Vegetation enhancement
    print(f"   Step 4/4: Vegetation enhancement...")
    final = enhance_vegetation(sharpened)

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
            height=final.shape[0],
            width=final.shape[1],
            count=3,
            dtype="uint8",
            crs=crs,
            transform=new_transform,
            compress="lzw",
        ) as dst:
            for i in range(3):
                dst.write(final[:, :, i], i + 1)

        print(f"   ✅ Saved GeoTIFF: {output_tif}")
        final_output = output_tif
    else:
        output_png = output_path.with_suffix(".png")
        cv2.imwrite(str(output_png), cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
        print(f"   ✅ Saved PNG: {output_png}")
        final_output = output_png

    # Also save PNG
    output_png = output_path.with_suffix(".png")
    cv2.imwrite(str(output_png), cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
    print(f"   ✅ Saved PNG: {output_png}")

    metadata = {
        "input_file": str(input_path),
        "output_file": str(final_output),
        "scale": actual_scale,
        "model": f"RealESRGAN_farm_x{actual_scale}",
        "enhancements": [
            "Real-ESRGAN super-resolution",
            "CLAHE local contrast",
            "Unsharp mask edge sharpening",
            "Vegetation enhancement",
        ],
        "original_size": list(original_shape),
        "output_size": list(final.shape[:2]),
        "original_resolution_m": 10.0,
        "optimized_for": "crop_row_visibility",
    }

    return final_output, metadata


def process_farm_sr(
    input_tif: Path,
    output_dir: Path,
    scale: int = 4,
) -> dict:
    """
    Process Sentinel-2 imagery with farm-optimized super-resolution.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(input_tif).stem
    sr_tif = output_dir / f"{base_name}_farm_sr_x{scale}.tif"

    output_path, sr_metadata = apply_farm_sr(
        input_path=input_tif,
        output_path=sr_tif,
        scale=scale,
    )

    result = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "input": str(input_tif),
        "outputs": {
            "sr_tif": str(sr_tif) if sr_tif.exists() else None,
            "sr_png": (
                str(sr_tif.with_suffix(".png"))
                if sr_tif.with_suffix(".png").exists()
                else None
            ),
        },
        "sr_metadata": sr_metadata,
    }

    meta_file = output_dir / f"{base_name}_farm_sr_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n🌾 Farm Super-Resolution complete!")
    print(f"   Optimized for: Crop row visibility")
    print(f"   Enhancements: Real-ESRGAN + CLAHE + Edge sharpening + Vegetation boost")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Farm-optimized Super-Resolution")
    parser.add_argument("input", help="Input GeoTIFF file")
    parser.add_argument(
        "-o", "--output", help="Output directory", default="./farm_sr_output"
    )
    parser.add_argument(
        "-s", "--scale", type=int, choices=[2, 4], default=4, help="SR scale"
    )

    args = parser.parse_args()

    result = process_farm_sr(
        input_tif=Path(args.input),
        output_dir=Path(args.output),
        scale=args.scale,
    )

    print(f"\nResults: {result['outputs']}")
