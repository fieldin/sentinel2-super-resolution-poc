"""
Skycues-like Multi-Temporal Super-Resolution Pipeline

Pipeline:
1. Collect 8-20 Sentinel-2 L2A scenes (14-30 days)
2. Aggressive cloud/shadow masking using SCL band
3. Sub-pixel alignment between scenes
4. Weighted temporal fusion (NOT just median)
5. SR model on fused result
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import cv2

try:
    import rasterio
    from rasterio.transform import Affine

    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

try:
    from pystac_client import Client

    PYSTAC_AVAILABLE = True
except ImportError:
    PYSTAC_AVAILABLE = False


STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"

SCL_BAD_VALUES = [0, 1, 3, 8, 9, 10, 11]


def search_scenes(
    geometry: dict,
    days_back: int = 30,
    max_scenes: int = 20,
    max_cloud_metadata: float = 50,
) -> List[dict]:
    """Collect many Sentinel-2 L2A scenes for temporal fusion."""
    if not PYSTAC_AVAILABLE:
        raise ImportError("pystac_client not installed")

    print(f"\n📡 Step 1: Collecting Sentinel-2 L2A scenes")
    print(f"   Time window: {days_back} days")
    print(f"   Target scenes: {max_scenes}")

    client = Client.open(STAC_URL)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)

    search = client.search(
        collections=[COLLECTION],
        intersects=geometry,
        datetime=f"{start_date.isoformat()}Z/{end_date.isoformat()}Z",
        query={"eo:cloud_cover": {"lte": max_cloud_metadata}},
        max_items=max_scenes * 2,
        sortby=[{"field": "properties.datetime", "direction": "desc"}],
    )

    items = list(search.items())
    print(f"   Found {len(items)} candidate scenes")

    scenes = []
    for item in items[:max_scenes]:
        view_angle = item.properties.get(
            "view:off_nadir", item.properties.get("s2:mean_solar_zenith_angle", 0)
        )

        scenes.append(
            {
                "id": item.id,
                "datetime": item.properties["datetime"],
                "cloud_cover": item.properties.get("eo:cloud_cover", 0),
                "view_angle": view_angle,
                "assets": {
                    "visual": (
                        item.assets.get("visual", {}).href
                        if "visual" in item.assets
                        else None
                    ),
                    "scl": (
                        item.assets.get("scl", {}).href
                        if "scl" in item.assets
                        else None
                    ),
                },
            }
        )
        print(
            f"   • {item.properties['datetime'][:10]} | "
            f"{item.properties.get('eo:cloud_cover', 0):.0f}% cloud | "
            f"angle: {view_angle:.1f}°"
        )

    return scenes


def download_and_mask_scene(
    scene: dict,
    geometry: dict,
    output_dir: Path,
) -> Optional[Tuple[np.ndarray, np.ndarray, dict]]:
    """Download scene, apply aggressive cloud/shadow masking using SCL."""
    import subprocess

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    visual_url = scene["assets"].get("visual")
    scl_url = scene["assets"].get("scl")

    if not visual_url:
        print(f"   ⚠️ No visual asset for {scene['id']}")
        return None

    cutline_file = output_dir / "cutline_temp.geojson"
    with open(cutline_file, "w") as f:
        json.dump(
            {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "properties": {}, "geometry": geometry}
                ],
            },
            f,
        )

    date_str = scene["datetime"][:10].replace("-", "")

    visual_file = output_dir / f"visual_{date_str}.tif"
    cmd = [
        "gdalwarp",
        "-overwrite",
        "-cutline",
        str(cutline_file),
        "-crop_to_cutline",
        "-t_srs",
        "EPSG:4326",
        "-co",
        "COMPRESS=LZW",
        f"/vsicurl/{visual_url}",
        str(visual_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0 or not visual_file.exists():
        print(f"   ❌ Failed to download visual for {date_str}")
        cutline_file.unlink(missing_ok=True)
        return None

    with rasterio.open(visual_file) as src:
        img = src.read()
        transform = src.transform
        crs = src.crs

    valid_mask = np.ones(img.shape[1:], dtype=bool)

    if scl_url:
        scl_file = output_dir / f"scl_{date_str}.tif"
        cmd = [
            "gdalwarp",
            "-overwrite",
            "-cutline",
            str(cutline_file),
            "-crop_to_cutline",
            "-t_srs",
            "EPSG:4326",
            "-ts",
            str(img.shape[2]),
            str(img.shape[1]),
            "-r",
            "nearest",
            "-co",
            "COMPRESS=LZW",
            f"/vsicurl/{scl_url}",
            str(scl_file),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0 and scl_file.exists():
            with rasterio.open(scl_file) as src:
                scl = src.read(1)

            for bad_val in SCL_BAD_VALUES:
                valid_mask &= scl != bad_val

            valid_pct = valid_mask.sum() / valid_mask.size * 100
            print(f"   ✓ {date_str}: {valid_pct:.1f}% valid pixels after SCL mask")

            scl_file.unlink(missing_ok=True)
        else:
            print(f"   ⚠️ No SCL for {date_str}, using all pixels")

    cutline_file.unlink(missing_ok=True)
    visual_file.unlink(missing_ok=True)

    img = img.astype(np.float32)
    if img.max() > 1:
        img = img / 255.0 if img.max() <= 255 else img / 10000.0

    return (
        img,
        valid_mask,
        {
            "transform": transform,
            "crs": crs,
            "datetime": scene["datetime"],
            "view_angle": scene["view_angle"],
            "cloud_cover": scene["cloud_cover"],
        },
    )


def align_scenes_subpixel(
    scenes: List[Tuple[np.ndarray, np.ndarray, dict]],
    reference_idx: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray, dict]]:
    """Align scenes at sub-pixel level using phase correlation."""
    print(f"\n🔧 Step 3: Sub-pixel alignment")
    print(f"   Reference: scene {reference_idx}")

    if len(scenes) < 2:
        return scenes

    ref_img, ref_mask, ref_info = scenes[reference_idx]

    if ref_img.shape[0] >= 3:
        ref_gray = 0.299 * ref_img[0] + 0.587 * ref_img[1] + 0.114 * ref_img[2]
    else:
        ref_gray = ref_img[0]

    aligned = [scenes[reference_idx]]

    for i, (img, mask, info) in enumerate(scenes):
        if i == reference_idx:
            continue

        if img.shape[0] >= 3:
            gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        else:
            gray = img[0]

        try:
            f_ref = np.fft.fft2(ref_gray)
            f_img = np.fft.fft2(gray)

            cross_power = (f_ref * np.conj(f_img)) / (
                np.abs(f_ref * np.conj(f_img)) + 1e-10
            )
            shift_map = np.abs(np.fft.ifft2(cross_power))

            max_loc = np.unravel_index(np.argmax(shift_map), shift_map.shape)

            dy = (
                max_loc[0]
                if max_loc[0] < shift_map.shape[0] // 2
                else max_loc[0] - shift_map.shape[0]
            )
            dx = (
                max_loc[1]
                if max_loc[1] < shift_map.shape[1] // 2
                else max_loc[1] - shift_map.shape[1]
            )

            if abs(dx) > 0.1 or abs(dy) > 0.1:
                M = np.float32([[1, 0, -dx], [0, 1, -dy]])

                aligned_img = np.zeros_like(img)
                for c in range(img.shape[0]):
                    aligned_img[c] = cv2.warpAffine(
                        img[c],
                        M,
                        (img.shape[2], img.shape[1]),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )

                aligned_mask = cv2.warpAffine(
                    mask.astype(np.uint8),
                    M,
                    (mask.shape[1], mask.shape[0]),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                ).astype(bool)

                print(f"   Scene {i}: shift ({dx:.2f}, {dy:.2f}) px")
                aligned.append((aligned_img, aligned_mask, info))
            else:
                print(f"   Scene {i}: aligned (shift < 0.1 px)")
                aligned.append((img, mask, info))

        except Exception as e:
            print(f"   Scene {i}: alignment failed ({e}), using original")
            aligned.append((img, mask, info))

    return aligned


def weighted_temporal_fusion(
    scenes: List[Tuple[np.ndarray, np.ndarray, dict]],
) -> Tuple[np.ndarray, dict]:
    """Weighted per-pixel temporal fusion."""
    print(f"\n🔬 Step 4: Weighted temporal fusion")
    print(f"   Scenes: {len(scenes)}")

    if not scenes:
        raise ValueError("No scenes to fuse")

    if len(scenes) == 1:
        return scenes[0][0], {"method": "single_scene"}

    ref_shape = scenes[0][0].shape

    weighted_sum = np.zeros(ref_shape, dtype=np.float64)
    weight_sum = np.zeros(ref_shape[1:], dtype=np.float64)

    for img, mask, info in scenes:
        if img.shape != ref_shape:
            continue

        view_angle = info.get("view_angle", 10)
        angle_weight = 1.0 / (1.0 + view_angle / 20.0)

        if img.shape[0] >= 3:
            gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        else:
            gray = img[0]

        laplacian = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
        variance_map = cv2.blur(np.abs(laplacian), (5, 5))
        variance_weight = variance_map / (variance_map.max() + 1e-10)

        local_mean = cv2.blur(gray.astype(np.float32), (15, 15))
        contrast = np.abs(gray - local_mean)
        contrast_weight = contrast / (contrast.max() + 1e-10)

        pixel_weight = (
            angle_weight * 0.3 + variance_weight * 0.4 + contrast_weight * 0.3
        )

        pixel_weight = pixel_weight * mask.astype(np.float32)

        for c in range(img.shape[0]):
            weighted_sum[c] += img[c] * pixel_weight
        weight_sum += pixel_weight

    weight_sum = np.maximum(weight_sum, 1e-10)
    fused = np.zeros(ref_shape, dtype=np.float32)
    for c in range(ref_shape[0]):
        fused[c] = weighted_sum[c] / weight_sum

    all_imgs = np.stack([s[0] for s in scenes], axis=0)

    no_data = weight_sum < 0.01
    if no_data.any():
        print(f"   Filling {no_data.sum()} gap pixels with median")
        for c in range(ref_shape[0]):
            channel_stack = all_imgs[:, c, :, :]
            median = np.median(channel_stack, axis=0)
            fused[c] = np.where(no_data, median, fused[c])

    valid_pct = (weight_sum > 0.01).sum() / weight_sum.size * 100
    print(f"   Fusion complete: {valid_pct:.1f}% coverage")

    return fused, {
        "method": "weighted_temporal_fusion",
        "scenes_used": len(scenes),
        "coverage_pct": valid_pct,
    }


def apply_sr_to_fused(
    fused: np.ndarray,
    scale: int = 2,
    model_type: str = "edsr",
) -> np.ndarray:
    """Apply super-resolution to the fused composite."""
    print(f"\n📐 Step 5: Super-Resolution (x{scale})")

    img_uint8 = (fused * 255).clip(0, 255).astype(np.uint8)

    if img_uint8.shape[0] <= 4:
        img_hwc = np.transpose(img_uint8, (1, 2, 0))
    else:
        img_hwc = img_uint8

    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()

        model_dir = Path(__file__).parent.parent / "models"
        model_dir.mkdir(exist_ok=True)

        model_name = f"EDSR_x{scale}.pb"
        model_path = model_dir / model_name

        if not model_path.exists():
            print(f"   📥 Downloading {model_name}...")
            import urllib.request

            url = f"https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/{model_name}"
            urllib.request.urlretrieve(url, model_path)

        sr.readModel(str(model_path))
        sr.setModel("edsr", scale)

        result = sr.upsample(img_hwc)
        print(f"   Output: {result.shape[1]}x{result.shape[0]}")

        result = np.transpose(result, (2, 0, 1)).astype(np.float32) / 255.0

        return result

    except Exception as e:
        print(f"   ⚠️ SR failed: {e}, using bicubic upscale")

        h, w = img_hwc.shape[:2]
        result = cv2.resize(
            img_hwc, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC
        )
        result = np.transpose(result, (2, 0, 1)).astype(np.float32) / 255.0

        return result


def skycues_pipeline(
    geometry: dict,
    output_dir: Path,
    days_back: int = 30,
    target_scenes: int = 15,
    sr_scale: int = 2,
) -> Dict[str, Any]:
    """Complete Skycues-like pipeline."""
    print("=" * 60)
    print("🛰️ Skycues-like Multi-Temporal SR Pipeline")
    print("=" * 60)
    print(f"   Target: ~70-85% of Skycues quality")
    print(f"   Cost: ~1-5% of commercial SR")
    print()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)

    scenes_meta = search_scenes(
        geometry,
        days_back=days_back,
        max_scenes=target_scenes,
    )

    if len(scenes_meta) < 3:
        return {"error": "Not enough scenes found (need at least 3)"}

    print(f"\n☁️ Step 2: Download & cloud mask")
    loaded_scenes = []
    for scene in scenes_meta:
        result = download_and_mask_scene(scene, geometry, temp_dir)
        if result is not None:
            img, mask, info = result
            valid_pct = mask.sum() / mask.size * 100
            if valid_pct > 30:
                loaded_scenes.append(result)

    print(f"   Usable scenes: {len(loaded_scenes)}/{len(scenes_meta)}")

    if len(loaded_scenes) < 2:
        return {"error": "Not enough usable scenes after masking"}

    aligned_scenes = align_scenes_subpixel(loaded_scenes)

    fused, fusion_info = weighted_temporal_fusion(aligned_scenes)

    sr_result = apply_sr_to_fused(fused, scale=sr_scale)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    ref_info = loaded_scenes[0][2]
    transform = ref_info["transform"]
    crs = ref_info["crs"]

    sr_transform = Affine(
        transform.a / sr_scale,
        transform.b,
        transform.c,
        transform.d,
        transform.e / sr_scale,
        transform.f,
    )

    fused_path = output_dir / f"fused_{timestamp}.tif"
    fused_uint8 = (fused * 255).clip(0, 255).astype(np.uint8)
    with rasterio.open(
        fused_path,
        "w",
        driver="GTiff",
        height=fused.shape[1],
        width=fused.shape[2],
        count=fused.shape[0],
        dtype="uint8",
        crs=crs,
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(fused_uint8)

    sr_path = output_dir / f"sr_x{sr_scale}_{timestamp}.tif"
    sr_uint8 = (sr_result * 255).clip(0, 255).astype(np.uint8)
    with rasterio.open(
        sr_path,
        "w",
        driver="GTiff",
        height=sr_result.shape[1],
        width=sr_result.shape[2],
        count=sr_result.shape[0],
        dtype="uint8",
        crs=crs,
        transform=sr_transform,
        compress="lzw",
    ) as dst:
        dst.write(sr_uint8)

    result = {
        "timestamp": timestamp,
        "scenes_collected": len(scenes_meta),
        "scenes_used": len(loaded_scenes),
        "fusion_method": fusion_info["method"],
        "sr_scale": sr_scale,
        "fused_path": str(fused_path),
        "sr_path": str(sr_path),
        "fused_size": [fused.shape[2], fused.shape[1]],
        "sr_size": [sr_result.shape[2], sr_result.shape[1]],
        "original_resolution_m": 10.0,
        "final_resolution_m": 10.0 / sr_scale,
    }

    meta_path = output_dir / f"pipeline_{timestamp}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n" + "=" * 60)
    print(f"✅ Pipeline Complete!")
    print("=" * 60)
    print(f"   Scenes fused: {len(loaded_scenes)}")
    print(f"   Final size: {sr_result.shape[2]}x{sr_result.shape[1]}")
    print(f"   Resolution: 10m → {10.0/sr_scale}m")
    print(f"   Output: {sr_path}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Skycues-like Multi-Temporal SR Pipeline"
    )
    parser.add_argument("-p", "--polygon", required=True, help="GeoJSON file or string")
    parser.add_argument(
        "-o", "--output", default="./skycues_output", help="Output directory"
    )
    parser.add_argument("-d", "--days", type=int, default=30, help="Days to look back")
    parser.add_argument("-n", "--scenes", type=int, default=15, help="Target scenes")
    parser.add_argument("-s", "--scale", type=int, default=2, choices=[2, 4])

    args = parser.parse_args()

    polygon_path = Path(args.polygon)
    if polygon_path.exists():
        with open(polygon_path) as f:
            data = json.load(f)
    else:
        data = json.loads(args.polygon)

    if data.get("type") == "FeatureCollection":
        geometry = data["features"][0]["geometry"]
    elif data.get("type") == "Feature":
        geometry = data["geometry"]
    else:
        geometry = data

    result = skycues_pipeline(
        geometry=geometry,
        output_dir=Path(args.output),
        days_back=args.days,
        target_scenes=args.scenes,
        sr_scale=args.scale,
    )

    print(f"\nDone! Output: {result.get('sr_path')}")
