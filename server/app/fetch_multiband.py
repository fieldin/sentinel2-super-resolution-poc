"""app.fetch_multiband

Fetch and stack Sentinel-2 L2A bands needed for real NDVI:
  - B04 (red, 10m)  -> EarthSearch asset key: "red"
  - B08 (NIR, 10m)  -> EarthSearch asset key: "nir"
  - SCL (Scene Classification Layer, 20m) -> asset key: "scl" (resampled to 10m)

Outputs a single GeoTIFF clipped to AOI:
  data/source/s2_multiband_<timestamp>_<scene_id>.tif

Bands (1-indexed) in output:
  1: B04 (uint16 reflectance, typically scaled by 10000)
  2: B08 (uint16 reflectance, typically scaled by 10000)
  3: SCL (uint16 class codes 0-11, resampled nearest)

Notes:
- Designed to be POC-friendly and runs via COG streaming (GDAL /vsicurl/).
- Output is reprojected to EPSG:4326 for Mapbox alignment.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import requests
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_geom

from .utils import ensure_directory, read_json, setup_logging

logger = setup_logging("fetch_multiband")

EARTHSEARCH_STAC_SEARCH = "https://earth-search.aws.element84.com/v1/search"


@dataclass
class FetchConfig:
    max_cloud_cover: float = 20.0
    max_age_days: int = 30
    limit: int = 5


def _extract_aoi_geom(aoi_path: Path) -> Dict[str, Any]:
    aoi_data = read_json(aoi_path)
    if aoi_data.get("type") == "FeatureCollection":
        return aoi_data["features"][0]["geometry"]
    if aoi_data.get("type") == "Feature":
        return aoi_data["geometry"]
    return aoi_data


def _stac_search(aoi_geom: Dict[str, Any], cfg: FetchConfig) -> Dict[str, Any]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=cfg.max_age_days)

    payload = {
        "collections": ["sentinel-2-l2a"],
        "intersects": aoi_geom,
        "datetime": f"{start.strftime('%Y-%m-%d')}T00:00:00Z/{end.strftime('%Y-%m-%d')}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lte": cfg.max_cloud_cover}},
        "limit": cfg.limit,
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    }

    r = requests.post(EARTHSEARCH_STAC_SEARCH, json=payload, timeout=45)
    r.raise_for_status()
    return r.json()


def _get_asset_href(assets: Dict[str, Any], key: str) -> Optional[str]:
    a = assets.get(key)
    return a.get("href") if a else None


def _open_vsicurl(href: str):
    return rasterio.open(f"/vsicurl/{href}")


def _clip_to_aoi(src: rasterio.io.DatasetReader, aoi_geom_wgs84: Dict[str, Any]) -> Tuple[np.ndarray, rasterio.Affine]:
    aoi_in_src = transform_geom("EPSG:4326", src.crs, aoi_geom_wgs84, precision=6)
    data, transform = rio_mask(src, [aoi_in_src], crop=True, filled=True)
    return data, transform


def fetch_and_stack_multiband(aoi_path: Path, out_dir: Path, cfg: FetchConfig) -> Path:
    ensure_directory(out_dir)
    aoi_geom = _extract_aoi_geom(aoi_path)

    logger.info("Searching Sentinel-2 L2A via EarthSearch STAC…")
    results = _stac_search(aoi_geom, cfg)
    features = results.get("features", [])
    if not features:
        raise RuntimeError("No Sentinel-2 L2A scenes found for AOI/date/cloud filters")

    scene = features[0]
    scene_id = scene.get("id", "unknown")
    props = scene.get("properties", {})
    assets = scene.get("assets", {})

    cloud = props.get("eo:cloud_cover", None)
    dt = props.get("datetime", "")

    red_href = _get_asset_href(assets, "red")
    nir_href = _get_asset_href(assets, "nir")
    scl_href = _get_asset_href(assets, "scl")

    if not red_href or not nir_href:
        raise RuntimeError("Scene is missing required assets (red/nir)")

    logger.info(f"Selected scene: {scene_id} cloud={cloud}% datetime={dt}")

    # Reference grid: clipped RED
    with _open_vsicurl(red_href) as red_src:
        red_data, ref_transform = _clip_to_aoi(red_src, aoi_geom)
        red_band = red_data[0].astype(np.uint16)

        ref_crs = red_src.crs
        ref_shape = red_band.shape

        # NIR → reference grid
        with _open_vsicurl(nir_href) as nir_src:
            nir_data, nir_transform = _clip_to_aoi(nir_src, aoi_geom)
            nir_raw = nir_data[0]
            nir_band = np.zeros(ref_shape, dtype=np.uint16)
            reproject(
                source=nir_raw,
                destination=nir_band,
                src_transform=nir_transform,
                src_crs=nir_src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear,
            )

        # SCL (20m) → reference grid (nearest)
        scl_band = np.zeros(ref_shape, dtype=np.uint16)
        if scl_href:
            with _open_vsicurl(scl_href) as scl_src:
                scl_data, scl_transform = _clip_to_aoi(scl_src, aoi_geom)
                scl_raw = scl_data[0]
                reproject(
                    source=scl_raw,
                    destination=scl_band,
                    src_transform=scl_transform,
                    src_crs=scl_src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.nearest,
                )

        tmp = out_dir / f"_tmp_s2_multiband_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tif"
        meta = red_src.meta.copy()
        meta.update(count=3, dtype="uint16")

        with rasterio.open(tmp, "w", **meta) as dst:
            dst.write(red_band, 1)
            dst.write(nir_band, 2)
            dst.write(scl_band, 3)

    # Reproject to EPSG:4326
    out_path = out_dir / f"s2_multiband_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{scene_id[:12]}.tif"
    with rasterio.open(tmp) as src:
        dst_crs = "EPSG:4326"
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        meta = src.meta.copy()
        meta.update(crs=dst_crs, transform=transform, width=width, height=height)

        with rasterio.open(out_path, "w", **meta) as dst:
            for b in (1, 2, 3):
                reproject(
                    source=rasterio.band(src, b),
                    destination=rasterio.band(dst, b),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest if b == 3 else Resampling.bilinear,
                )

    try:
        tmp.unlink()
    except Exception:
        pass

    logger.info(f"Saved multiband stack: {out_path}")
    logger.info("Bands: 1=B04(red) 2=B08(nir) 3=SCL (uint16)")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch & stack Sentinel-2 L2A bands (B04,B08,SCL) clipped to AOI")
    p.add_argument("--aoi", required=True, help="Path to AOI GeoJSON (EPSG:4326)")
    p.add_argument("--out", default="data/source", help="Output directory (default: data/source)")
    p.add_argument("--max-cloud", type=float, default=20.0, help="Max cloud cover percent (default: 20)")
    p.add_argument("--days", type=int, default=30, help="Lookback days (default: 30)")
    p.add_argument("--limit", type=int, default=5, help="Max scenes to consider (default: 5)")
    args = p.parse_args()

    out = fetch_and_stack_multiband(
        aoi_path=Path(args.aoi),
        out_dir=Path(args.out),
        cfg=FetchConfig(max_cloud_cover=args.max_cloud, max_age_days=args.days, limit=args.limit),
    )
    print(str(out))


if __name__ == "__main__":
    main()
