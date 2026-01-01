"""
True Super-Resolution module for Sentinel-2 imagery.
Uses OpenCV DNN with EDSR/ESPCN models for reliable SR without GPU dependencies.
"""

import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Optional, Tuple
import json
from datetime import datetime

import numpy as np
import cv2
import rasterio
from rasterio.transform import Affine


# Model URLs for OpenCV SR
SR_MODELS = {
    "edsr_x2": {
        "url": "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x2.pb",
        "scale": 2,
    },
    "edsr_x3": {
        "url": "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x3.pb",
        "scale": 3,
    },
    "edsr_x4": {
        "url": "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb",
        "scale": 4,
    },
    "espcn_x2": {
        "url": "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x2.pb",
        "scale": 2,
    },
    "espcn_x3": {
        "url": "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x3.pb",
        "scale": 3,
    },
    "espcn_x4": {
        "url": "https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x4.pb",
        "scale": 4,
    },
    "lapsrn_x2": {
        "url": "https://github.com/fannymonori/TF-LapSRN/raw/master/export/LapSRN_x2.pb",
        "scale": 2,
    },
    "lapsrn_x4": {
        "url": "https://github.com/fannymonori/TF-LapSRN/raw/master/export/LapSRN_x4.pb",
        "scale": 4,
    },
    "lapsrn_x8": {
        "url": "https://github.com/fannymonori/TF-LapSRN/raw/master/export/LapSRN_x8.pb",
        "scale": 8,
    },
}


def get_model_dir() -> Path:
    """Get or create model directory."""
    model_dir = Path(__file__).parent.parent / "models"
    model_dir.mkdir(exist_ok=True)
    return model_dir


def download_model(model_name: str) -> Path:
    """Download SR model if not present."""
    if model_name not in SR_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(SR_MODELS.keys())}"
        )

    model_info = SR_MODELS[model_name]
    model_dir = get_model_dir()
    model_path = model_dir / f"{model_name}.pb"

    if not model_path.exists():
        print(f"📥 Downloading {model_name} model...")
        try:
            urllib.request.urlretrieve(model_info["url"], model_path)
            print(f"   ✅ Saved to {model_path}")
        except Exception as e:
            print(f"   ❌ Failed to download: {e}")
            raise

    return model_path


def create_sr_model(scale: int = 4, model_type: str = "edsr"):
    """
    Create OpenCV DNN Super-Resolution model.

    Args:
        scale: Upscaling factor (2, 3, 4, or 8 for lapsrn)
        model_type: Model type ('edsr', 'espcn', or 'lapsrn')

    Returns:
        OpenCV DNN Super Resolution object
    """
    # Select model name
    model_name = f"{model_type}_x{scale}"

    if model_name not in SR_MODELS:
        # Fallback to available scales
        available = [k for k in SR_MODELS.keys() if k.startswith(model_type)]
        if not available:
            model_type = "edsr"
            model_name = f"edsr_x{min(scale, 4)}"
        else:
            model_name = available[0]
        print(f"   ⚠️  Using {model_name} instead")

    model_path = download_model(model_name)
    actual_scale = SR_MODELS[model_name]["scale"]

    # Create OpenCV SR object
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(model_path))
    sr.setModel(model_type, actual_scale)

    return sr, actual_scale


def apply_super_resolution(
    input_path: Path,
    output_path: Path,
    scale: int = 4,
    model_type: str = "edsr",
    output_format: str = "tif",
) -> Tuple[Path, dict]:
    """
    Apply Deep Learning super-resolution to an image.

    Args:
        input_path: Path to input image (GeoTIFF or PNG)
        output_path: Path for output image
        scale: Upscaling factor (2, 3, or 4)
        model_type: Model type ('edsr', 'espcn', 'lapsrn')
        output_format: Output format ('png' or 'tif')

    Returns:
        Tuple of (output_path, metadata_dict)
    """
    print(f"\n🔬 Super-Resolution x{scale} ({model_type.upper()})")
    print(f"   Input: {input_path}")

    # Read input image
    input_path = Path(input_path)
    transform = None
    crs = None

    if input_path.suffix.lower() in [".tif", ".tiff"]:
        # Read GeoTIFF with geospatial info
        with rasterio.open(input_path) as src:
            # Read RGB bands (assuming bands 1,2,3 are R,G,B)
            if src.count >= 3:
                img = np.stack([src.read(i) for i in [1, 2, 3]], axis=-1)
            else:
                img = src.read(1)
                img = np.stack([img, img, img], axis=-1)

            # Normalize to uint8 if needed
            if img.dtype != np.uint8:
                # Handle different data types
                if img.max() > 255:
                    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(
                        np.uint8
                    )
                else:
                    img = img.astype(np.uint8)

            # Save geospatial info
            transform = src.transform
            crs = src.crs
            original_shape = img.shape[:2]
    else:
        # Read regular image
        img = cv2.imread(str(input_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_shape = img.shape[:2]

    print(f"   Input size: {img.shape[1]}x{img.shape[0]} pixels")

    # Create SR model
    print(f"   Loading {model_type.upper()} x{scale} model...")
    sr_model, actual_scale = create_sr_model(scale=scale, model_type=model_type)

    # Convert to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Apply SR
    print(f"   Applying super-resolution (this may take a moment)...")
    output_bgr = sr_model.upsample(img_bgr)
    output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)

    print(f"   Output size: {output_rgb.shape[1]}x{output_rgb.shape[0]} pixels")

    # Save output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "tif" and transform is not None:
        # Save as GeoTIFF with updated transform
        new_transform = Affine(
            transform.a / actual_scale,  # Pixel width (reduced)
            transform.b,
            transform.c,
            transform.d,
            transform.e / actual_scale,  # Pixel height (reduced)
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
        # Save as PNG
        output_png = output_path.with_suffix(".png")
        cv2.imwrite(str(output_png), cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR))
        print(f"   ✅ Saved PNG: {output_png}")
        final_output = output_png

    # Calculate effective resolution
    # Sentinel-2 native: 10m/pixel
    original_res = 10.0  # meters
    effective_res = original_res / actual_scale

    metadata = {
        "input_file": str(input_path),
        "output_file": str(final_output),
        "scale": actual_scale,
        "model": f"{model_type}_x{actual_scale}",
        "original_size": list(original_shape),
        "output_size": list(output_rgb.shape[:2]),
        "original_resolution_m": original_res,
        "effective_resolution_m": effective_res,
    }

    return final_output, metadata


def process_sentinel2_sr(
    input_tif: Path,
    output_dir: Path,
    scale: int = 4,
    model_type: str = "edsr",
) -> dict:
    """
    Process Sentinel-2 imagery with super-resolution.

    Args:
        input_tif: Path to Sentinel-2 GeoTIFF
        output_dir: Output directory
        scale: SR scale factor (2, 3, or 4)
        model_type: Model type ('edsr', 'espcn', 'lapsrn')

    Returns:
        Dictionary with output paths and metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(input_tif).stem

    # Output paths
    sr_tif = output_dir / f"{base_name}_sr_x{scale}.tif"
    sr_png = output_dir / f"{base_name}_sr_x{scale}.png"

    # Apply SR
    output_path, sr_metadata = apply_super_resolution(
        input_path=input_tif,
        output_path=sr_tif,
        scale=scale,
        model_type=model_type,
        output_format="tif",
    )

    # Also save PNG version
    if output_path.suffix == ".tif":
        with rasterio.open(output_path) as src:
            img = np.stack([src.read(i) for i in [1, 2, 3]], axis=-1)
        cv2.imwrite(str(sr_png), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"   ✅ Saved PNG: {sr_png}")

    # Save metadata
    result = {
        "timestamp": timestamp,
        "input": str(input_tif),
        "outputs": {
            "sr_tif": str(sr_tif) if sr_tif.exists() else None,
            "sr_png": str(sr_png) if sr_png.exists() else None,
        },
        "sr_metadata": sr_metadata,
    }

    meta_file = output_dir / f"{base_name}_sr_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✅ Super-Resolution complete!")
    print(f"   Model: {sr_metadata['model']}")
    print(f"   Effective resolution: {sr_metadata['effective_resolution_m']}m")

    return result


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply Super-Resolution to Sentinel-2 imagery"
    )
    parser.add_argument("input", help="Input GeoTIFF or PNG file")
    parser.add_argument(
        "-o", "--output", help="Output directory", default="./sr_output"
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=int,
        choices=[2, 3, 4],
        default=4,
        help="SR scale (2, 3, or 4)",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=["edsr", "espcn", "lapsrn"],
        default="edsr",
        help="SR model type",
    )

    args = parser.parse_args()

    result = process_sentinel2_sr(
        input_tif=Path(args.input),
        output_dir=Path(args.output),
        scale=args.scale,
        model_type=args.model,
    )

    print(f"\nResults: {result['outputs']}")
