"""
Smart Fetch: Automatically get the best Sentinel-2 image.

Strategy:
1. Check local files for recent images (last 30 days)
2. Check remote catalog for the best available image
3. Fetch from UP42/AWS only if remote is better or no local exists

Always selects: Latest + Clearest (lowest cloud cover) from last 30 days.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

from .settings import get_settings
from .utils import setup_logging, read_json, ensure_directory

logger = setup_logging("smart-fetch")


def get_local_images(source_dir: Path) -> list[dict]:
    """
    Get list of local images with their metadata.

    Returns:
        List of dicts with {path, scene_id, acquisition_date, cloud_cover, metadata}
    """
    images = []

    if not source_dir.exists():
        return images

    # Find all .tif files and their corresponding metadata
    for tif_file in source_dir.glob("*.tif"):
        # Look for matching metadata file
        meta_patterns = [
            tif_file.with_suffix(".json"),
            tif_file.parent / f"{tif_file.stem.rsplit('_', 1)[0]}_meta.json",
            tif_file.parent / f"{tif_file.stem}_meta.json",
        ]

        metadata = None
        for meta_path in meta_patterns:
            if meta_path.exists():
                try:
                    metadata = read_json(meta_path)
                    break
                except Exception:
                    continue

        if metadata:
            # Parse acquisition date
            acq_date_str = metadata.get("acquisition_date", "")
            try:
                if acq_date_str:
                    if acq_date_str.endswith("Z"):
                        acq_date_str = acq_date_str[:-1] + "+00:00"
                    acq_date = datetime.fromisoformat(
                        acq_date_str.replace("Z", "+00:00")
                    )
                else:
                    acq_date = datetime.fromtimestamp(tif_file.stat().st_mtime)
            except Exception:
                acq_date = datetime.fromtimestamp(tif_file.stat().st_mtime)

            images.append(
                {
                    "path": tif_file,
                    "scene_id": metadata.get("scene_id", tif_file.stem),
                    "acquisition_date": acq_date,
                    "cloud_cover": float(metadata.get("cloud_cover_pct", 100)),
                    "metadata": metadata,
                }
            )
        else:
            # No metadata, use file modification time
            images.append(
                {
                    "path": tif_file,
                    "scene_id": tif_file.stem,
                    "acquisition_date": datetime.fromtimestamp(
                        tif_file.stat().st_mtime
                    ),
                    "cloud_cover": 100.0,  # Unknown, assume worst
                    "metadata": None,
                }
            )

    return images


def select_best_local_image(
    source_dir: Path,
    max_age_days: int = 30,
    max_cloud_cover: float = 30.0,
) -> Optional[dict]:
    """
    Select the best local image: most recent with lowest cloud cover.

    Args:
        source_dir: Directory containing source images
        max_age_days: Maximum age of image in days
        max_cloud_cover: Maximum cloud cover percentage

    Returns:
        Best image dict or None if no suitable image found
    """
    images = get_local_images(source_dir)

    if not images:
        logger.info("No local images found")
        return None

    # Filter by age and cloud cover
    cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)

    valid_images = []
    for img in images:
        # Make cutoff_date timezone-naive for comparison
        acq_date = img["acquisition_date"]
        if acq_date.tzinfo is not None:
            acq_date = acq_date.replace(tzinfo=None)

        if acq_date >= cutoff_date and img["cloud_cover"] <= max_cloud_cover:
            valid_images.append(img)

    if not valid_images:
        logger.info(
            f"No local images within {max_age_days} days with cloud <= {max_cloud_cover}%"
        )
        return None

    # Sort by cloud cover (ascending), then by date (descending = newest first)
    valid_images.sort(
        key=lambda x: (x["cloud_cover"], -x["acquisition_date"].timestamp())
    )

    best = valid_images[0]
    logger.info(
        f"Best local image: {best['scene_id']} "
        f"(cloud: {best['cloud_cover']}%, date: {best['acquisition_date'].date()})"
    )

    return best


def check_remote_catalog(
    aoi_geometry: dict,
    max_age_days: int = 30,
    max_cloud_cover: float = 30.0,
) -> Optional[dict]:
    """
    Check remote catalog (AWS Earth Search) for best available image.

    Returns:
        Best remote scene dict or None if search fails
    """
    import requests

    stac_url = "https://earth-search.aws.element84.com/v1/search"

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=max_age_days)

    search_payload = {
        "collections": ["sentinel-2-l2a"],
        "intersects": aoi_geometry,
        "datetime": f"{start_date.strftime('%Y-%m-%d')}T00:00:00Z/{end_date.strftime('%Y-%m-%d')}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lte": max_cloud_cover}},
        "limit": 10,
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    }

    try:
        response = requests.post(
            stac_url,
            json=search_payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        results = response.json()
    except Exception as e:
        logger.warning(f"Remote catalog search failed: {e}")
        return None

    features = results.get("features", [])
    if not features:
        logger.info("No remote images found matching criteria")
        return None

    # Get best (already sorted by cloud cover)
    best = features[0]
    props = best.get("properties", {})

    acq_date_str = props.get("datetime", "")
    if acq_date_str:
        if acq_date_str.endswith("Z"):
            acq_date_str = acq_date_str[:-1] + "+00:00"
        acq_date = datetime.fromisoformat(acq_date_str.replace("Z", "+00:00"))
    else:
        acq_date = datetime.utcnow()

    result = {
        "scene_id": best.get("id", "unknown"),
        "acquisition_date": acq_date,
        "cloud_cover": props.get("eo:cloud_cover", 100),
        "feature": best,
    }

    logger.info(
        f"Best remote image: {result['scene_id']} "
        f"(cloud: {result['cloud_cover']}%, date: {result['acquisition_date'].date()})"
    )

    return result


def smart_fetch(
    aoi_geometry: dict,
    source_dir: Path,
    max_age_days: int = 30,
    max_cloud_cover: float = 30.0,
    force_fetch: bool = False,
) -> Tuple[Path, dict]:
    """
    Smart fetch: Get best available Sentinel-2 image.

    Strategy:
    1. Check local for recent images
    2. Check remote catalog
    3. Fetch only if:
       - No local image exists
       - Remote has significantly better image (lower cloud or newer)
       - force_fetch is True

    Args:
        aoi_geometry: GeoJSON geometry for AOI
        source_dir: Directory for source images
        max_age_days: Maximum age in days (default 30)
        max_cloud_cover: Maximum cloud cover % (default 30)
        force_fetch: Force fetch even if local is good

    Returns:
        Tuple of (path to best GeoTIFF, metadata dict)
    """
    logger.info("=" * 60)
    logger.info("Smart Fetch: Finding best Sentinel-2 image")
    logger.info(f"  Max age: {max_age_days} days")
    logger.info(f"  Max cloud: {max_cloud_cover}%")
    logger.info("=" * 60)

    ensure_directory(source_dir)

    # Step 1: Check local images
    best_local = select_best_local_image(source_dir, max_age_days, max_cloud_cover)

    # Step 2: Check remote catalog
    best_remote = check_remote_catalog(aoi_geometry, max_age_days, max_cloud_cover)

    # Step 3: Decide whether to fetch
    should_fetch = False
    reason = ""

    if force_fetch:
        should_fetch = True
        reason = "force_fetch=True"
    elif best_local is None:
        should_fetch = True
        reason = "No suitable local image"
    elif best_remote is not None:
        # Compare local vs remote
        local_cloud = best_local["cloud_cover"]
        remote_cloud = best_remote["cloud_cover"]

        local_date = best_local["acquisition_date"]
        remote_date = best_remote["acquisition_date"]

        # Make dates comparable (both naive)
        if local_date.tzinfo is not None:
            local_date = local_date.replace(tzinfo=None)
        if remote_date.tzinfo is not None:
            remote_date = remote_date.replace(tzinfo=None)

        # Fetch if remote is significantly better
        # Better = lower cloud cover OR newer with similar cloud
        if remote_cloud < local_cloud - 5:  # >5% better cloud cover
            should_fetch = True
            reason = (
                f"Remote has better cloud cover ({remote_cloud}% vs {local_cloud}%)"
            )
        elif (
            remote_date > local_date + timedelta(days=3) and remote_cloud <= local_cloud
        ):
            should_fetch = True
            reason = f"Remote is newer ({remote_date.date()} vs {local_date.date()})"
        elif best_local["scene_id"] == best_remote["scene_id"]:
            # Same scene, no need to fetch
            reason = "Same scene already local"

    # Step 4: Fetch if needed
    if should_fetch and best_remote is not None:
        logger.info(f"📥 Fetching from remote: {reason}")

        # Use the existing PublicSentinel2Client to download
        from .up42_client import PublicSentinel2Client

        settings = get_settings()
        client = PublicSentinel2Client(settings)

        output_path, metadata = client.fetch_best_scene(aoi_geometry, source_dir)

        logger.info(f"✅ Fetched: {output_path}")
        return output_path, metadata

    elif best_local is not None:
        logger.info(f"✅ Using local image: {best_local['path'].name}")
        logger.info(f"   Reason: {reason or 'Local is optimal'}")

        return best_local["path"], best_local["metadata"] or {
            "scene_id": best_local["scene_id"],
            "acquisition_date": best_local["acquisition_date"].isoformat(),
            "cloud_cover_pct": best_local["cloud_cover"],
            "file_path": str(best_local["path"]),
            "source": "local",
        }

    else:
        raise ValueError(
            f"No Sentinel-2 images available: "
            f"No local images within {max_age_days} days with cloud <= {max_cloud_cover}%, "
            f"and remote catalog search failed or returned no results."
        )


def ensure_best_image(
    source_dir: Optional[Path] = None,
    max_age_days: int = 30,
    max_cloud_cover: float = 30.0,
    force_fetch: bool = False,
) -> Tuple[Path, dict]:
    """
    Convenience function: ensure we have the best image available.

    Reads AOI from config and calls smart_fetch.

    Args:
        source_dir: Override source directory (default from settings)
        max_age_days: Maximum age in days
        max_cloud_cover: Maximum cloud cover %
        force_fetch: Force fetch even if local is good

    Returns:
        Tuple of (path to best GeoTIFF, metadata dict)
    """
    settings = get_settings()

    if source_dir is None:
        source_dir = Path(settings.data_dir) / "source"

    # Load AOI
    aoi_path = Path(settings.aoi_path)
    if not aoi_path.exists():
        raise FileNotFoundError(f"AOI file not found: {aoi_path}")

    aoi_data = read_json(aoi_path)

    # Extract geometry
    if aoi_data.get("type") == "FeatureCollection":
        aoi_geometry = aoi_data["features"][0]["geometry"]
    elif aoi_data.get("type") == "Feature":
        aoi_geometry = aoi_data["geometry"]
    else:
        aoi_geometry = aoi_data

    return smart_fetch(
        aoi_geometry=aoi_geometry,
        source_dir=source_dir,
        max_age_days=max_age_days,
        max_cloud_cover=max_cloud_cover,
        force_fetch=force_fetch,
    )


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Smart Fetch: Get best Sentinel-2 image"
    )
    parser.add_argument("--max-days", type=int, default=30, help="Max age in days")
    parser.add_argument(
        "--max-cloud", type=float, default=30.0, help="Max cloud cover %"
    )
    parser.add_argument("--force", action="store_true", help="Force fetch from remote")

    args = parser.parse_args()

    try:
        path, metadata = ensure_best_image(
            max_age_days=args.max_days,
            max_cloud_cover=args.max_cloud,
            force_fetch=args.force,
        )

        print(f"\n✅ Best image: {path}")
        print(f"   Scene ID: {metadata.get('scene_id', 'N/A')}")
        print(f"   Date: {metadata.get('acquisition_date', 'N/A')}")
        print(f"   Cloud: {metadata.get('cloud_cover_pct', 'N/A')}%")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
