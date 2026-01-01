#!/usr/bin/env python3
"""
Fetch cloud-free Sentinel-2 Surface Reflectance (L2A) imagery for a polygon.

Supports two modes:
1. UP42 API (requires credentials) - checks for existing orders first
2. AWS Earth Search (free, public) - direct download

Usage:
    # Using UP42 (checks existing orders first)
    python fetch_sentinel2_sr.py --polygon polygon.geojson --output ./output --use-up42

    # Using free AWS Earth Search (default)
    python fetch_sentinel2_sr.py --polygon polygon.geojson --output ./output

Requirements:
    pip install pystac-client rasterio numpy requests shapely
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
import subprocess

try:
    from pystac_client import Client
    from shapely.geometry import shape, box
    import requests
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pystac-client shapely requests")
    sys.exit(1)


# AWS Earth Search - Free public STAC catalog for Sentinel-2
STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"  # Level 2A = Surface Reflectance

# UP42 Configuration
UP42_AUTH_URL = "https://auth.up42.com/realms/public/protocol/openid-connect/token"
UP42_API_URL = "https://api.up42.com"
UP42_SENTINEL2_PRODUCT_ID = "c3de9ed8-f6e5-4bb5-a157-f6430ba756da"


class UP42Client:
    """UP42 API client with order caching."""

    def __init__(self, username: str, password: str, workspace_id: str):
        self.username = username
        self.password = password
        self.workspace_id = workspace_id
        self.token = None
        self.token_expires = None

    def authenticate(self):
        """Get or refresh access token."""
        if self.token and self.token_expires and datetime.utcnow() < self.token_expires:
            return self.token

        response = requests.post(
            UP42_AUTH_URL,
            data={
                "grant_type": "password",
                "username": self.username,
                "password": self.password,
                "client_id": "up42-api",
            },
        )
        response.raise_for_status()
        data = response.json()
        self.token = data["access_token"]
        self.token_expires = datetime.utcnow() + timedelta(
            seconds=data.get("expires_in", 300) - 30
        )
        return self.token

    @property
    def headers(self):
        return {
            "Authorization": f"Bearer {self.authenticate()}",
            "Content-Type": "application/json",
        }

    def get_existing_orders(self) -> list:
        """Get all existing orders."""
        response = requests.get(
            f"{UP42_API_URL}/v2/orders?workspaceId={self.workspace_id}",
            headers=self.headers,
        )
        if response.ok:
            return response.json().get("content", [])
        return []

    def find_existing_order(self, scene_id: str) -> dict:
        """Check if an order already exists for this scene."""
        orders = self.get_existing_orders()

        for order in orders:
            order_scene = order.get("orderDetails", {}).get("imageId", "")
            if scene_id in order_scene or order_scene in scene_id:
                return order

        return None

    def get_order_assets(self, order_id: str) -> list:
        """Get assets for a fulfilled order."""
        response = requests.get(
            f"{UP42_API_URL}/v2/orders/{order_id}/assets?workspaceId={self.workspace_id}",
            headers=self.headers,
        )
        if response.ok:
            return response.json().get("content", [])
        return []

    def search_catalog(
        self, bbox: list, start_date: datetime, end_date: datetime
    ) -> list:
        """Search UP42 catalog for Sentinel-2 scenes."""
        search_payload = {
            "collections": ["sentinel-2"],
            "bbox": bbox,
            "datetime": f"{start_date.strftime('%Y-%m-%d')}T00:00:00Z/{end_date.strftime('%Y-%m-%d')}T23:59:59Z",
            "limit": 20,
        }

        response = requests.post(
            f"{UP42_API_URL}/catalog/hosts/earthsearch-aws/stac/search",
            headers=self.headers,
            json=search_payload,
        )

        if response.ok:
            return response.json().get("features", [])
        return []

    def create_order(self, scene_id: str, aoi: dict) -> dict:
        """Create a new order for a scene."""
        order_payload = {
            "displayName": f"Sentinel2-SR-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "dataProduct": UP42_SENTINEL2_PRODUCT_ID,
            "featureCollection": aoi,
            "params": {"id": scene_id},
        }

        response = requests.post(
            f"{UP42_API_URL}/v2/orders?workspaceId={self.workspace_id}",
            headers=self.headers,
            json=order_payload,
        )

        return response.json()

    def wait_for_order(self, order_id: str, timeout_minutes: int = 15) -> dict:
        """Wait for order to be fulfilled."""
        max_attempts = timeout_minutes * 12  # Check every 5 seconds

        for i in range(max_attempts):
            response = requests.get(
                f"{UP42_API_URL}/v2/orders/{order_id}?workspaceId={self.workspace_id}",
                headers=self.headers,
            )

            if response.ok:
                order = response.json()
                status = order.get("status")

                if status == "FULFILLED":
                    return order
                elif status in ["FAILED", "CANCELLED", "FAILED_PERMANENTLY"]:
                    raise Exception(f"Order failed with status: {status}")

                if i % 12 == 0:  # Print every minute
                    print(f"   [{i // 12} min] Status: {status}")

            time.sleep(5)

        raise Exception("Order timeout")

    def download_asset(self, asset_id: str, output_path: Path) -> Path:
        """Download an asset to local file."""
        # Get download URL
        response = requests.get(
            f"{UP42_API_URL}/v2/assets/{asset_id}/downloadUrl?workspaceId={self.workspace_id}",
            headers=self.headers,
        )

        if not response.ok:
            raise Exception(f"Failed to get download URL: {response.text}")

        download_url = response.json().get("url")

        # Download file
        print(f"   Downloading asset {asset_id[:8]}...")
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        return output_path


def load_polygon(polygon_input: str) -> dict:
    """Load polygon from file path or JSON string."""
    if Path(polygon_input).exists():
        with open(polygon_input) as f:
            data = json.load(f)
    else:
        try:
            data = json.loads(polygon_input)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid polygon input: {polygon_input}")

    if data.get("type") == "FeatureCollection":
        geometry = data["features"][0]["geometry"]
    elif data.get("type") == "Feature":
        geometry = data["geometry"]
    elif data.get("type") in ["Polygon", "MultiPolygon"]:
        geometry = data
    else:
        raise ValueError(f"Unsupported GeoJSON type: {data.get('type')}")

    return geometry, data


def get_bbox(geometry: dict) -> list:
    """Calculate bounding box from geometry."""
    coords = geometry["coordinates"][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return [min(lons), min(lats), max(lons), max(lats)]


def fetch_via_up42(
    geometry: dict, aoi_data: dict, output_dir: Path, days_back: int, max_cloud: float
) -> Path:
    """Fetch imagery via UP42 API, reusing existing orders if available."""

    # Get credentials from environment
    username = os.getenv("UP42_USERNAME")
    password = os.getenv("UP42_PASSWORD")
    workspace_id = os.getenv("UP42_PROJECT_ID") or os.getenv("UP42_WORKSPACE_ID")

    if not all([username, password, workspace_id]):
        print("❌ UP42 credentials not found. Set environment variables:")
        print("   UP42_USERNAME, UP42_PASSWORD, UP42_PROJECT_ID")
        sys.exit(1)

    client = UP42Client(username, password, workspace_id)

    print("\n🔑 Authenticating with UP42...")
    client.authenticate()
    print("   ✅ Authenticated")

    # Search for scenes
    print("\n🔍 Searching UP42 catalog...")
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)
    bbox = get_bbox(geometry)

    scenes = client.search_catalog(bbox, start_date, end_date)
    print(f"   Found {len(scenes)} scenes")

    if not scenes:
        print("❌ No scenes found")
        return None

    # Extract scene info and sort by cloud cover
    scene_list = []
    for s in scenes:
        props = s.get("properties", {})
        scene_id = props.get("id")
        # UP42 uses 'cloudCoverage' at top level
        cloud = props.get("cloudCoverage", props.get("eo:cloud_cover", 100))
        date = props.get("datetime", "")[:10]

        print(f"   - {scene_id}: {cloud:.2f}% cloud, {date}")

        if cloud <= max_cloud:
            scene_list.append({"id": scene_id, "cloud": cloud, "date": date})

    scene_list.sort(key=lambda x: x["cloud"])

    if not scene_list:
        print(f"❌ No scenes with <= {max_cloud}% cloud cover")
        return None

    best = scene_list[0]
    print(f"\n✅ Best scene: {best['id']}")
    print(f"   Date: {best['date']}, Cloud: {best['cloud']:.2f}%")

    # Check for existing order
    print("\n📦 Checking for existing orders...")
    existing_order = client.find_existing_order(best["id"])

    if existing_order:
        order_id = existing_order["id"]
        status = existing_order["status"]
        print(f"   Found existing order: {order_id[:8]}...")
        print(f"   Status: {status}")

        if status == "FULFILLED":
            print("   ✅ Order already fulfilled! Reusing assets...")
            assets = client.get_order_assets(order_id)

            if assets:
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = (
                    output_dir
                    / f"sentinel2_sr_{best['date'].replace('-', '')}_{best['id'].split('_')[1]}.tif"
                )

                # Download first asset
                asset = assets[0]
                asset_id = asset.get("id")
                print(f"\n📥 Downloading from existing order...")
                client.download_asset(asset_id, output_file)

                print(f"\n✅ Saved: {output_file}")
                return output_file

        elif status == "BEING_FULFILLED":
            print("   ⏳ Order in progress, waiting...")
            order = client.wait_for_order(order_id)
            assets = client.get_order_assets(order_id)

            if assets:
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = (
                    output_dir
                    / f"sentinel2_sr_{best['date'].replace('-', '')}_{best['id'].split('_')[1]}.tif"
                )
                asset = assets[0]
                client.download_asset(asset["id"], output_file)
                print(f"\n✅ Saved: {output_file}")
                return output_file

    # No existing order - create new one
    print("\n📝 Creating new UP42 order...")
    result = client.create_order(best["id"], aoi_data)

    if result.get("results"):
        order_id = result["results"][0].get("id")
        print(f"   Order ID: {order_id}")

        print("\n⏳ Waiting for order to be fulfilled...")
        order = client.wait_for_order(order_id)

        assets = client.get_order_assets(order_id)
        if assets:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = (
                output_dir
                / f"sentinel2_sr_{best['date'].replace('-', '')}_{best['id'].split('_')[1]}.tif"
            )
            asset = assets[0]
            client.download_asset(asset["id"], output_file)
            print(f"\n✅ Saved: {output_file}")
            return output_file

    print(f"❌ Order failed: {result}")
    return None


def fetch_via_aws(
    geometry: dict,
    output_dir: Path,
    days_back: int,
    max_cloud: float,
    bands: list = None,
) -> dict:
    """Fetch imagery via free AWS Earth Search.

    Downloads only: visual, data (if available)
    Returns dict with paths to downloaded files.
    """

    # Assets to download for Sentinel-2 L2A (Surface Reflectance)
    # visual = RGB composite, scl = Scene Classification Layer
    WANTED_ASSETS = ["visual", "scl"]

    # Multispectral bands for SR analysis
    MULTISPECTRAL_BANDS = ["red", "green", "blue", "nir"]

    print(f"\n🔍 Searching AWS Earth Search for Sentinel-2 L2A...")
    print(f"   Days back: {days_back}")
    print(f"   Max cloud cover: {max_cloud}%")

    client = Client.open(STAC_URL)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)

    search = client.search(
        collections=[COLLECTION],
        intersects=geometry,
        datetime=f"{start_date.isoformat()}Z/{end_date.isoformat()}Z",
        query={"eo:cloud_cover": {"lte": max_cloud}},
        max_items=50,
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    )

    items = list(search.items())
    print(f"   Found {len(items)} scenes with <= {max_cloud}% cloud cover")

    if not items:
        print("\n❌ No cloud-free scenes found")
        return None

    # Select best
    best = items[0]
    print(f"\n✅ Selected scene:")
    print(f"   ID: {best.id}")
    print(f"   Date: {best.properties['datetime'][:10]}")
    print(f"   Cloud cover: {best.properties.get('eo:cloud_cover', 'N/A')}%")

    # Show available assets
    assets = best.assets
    print(f"\n📋 Available assets: {list(assets.keys())}")

    # Download
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_date = best.properties["datetime"][:10].replace("-", "")
    scene_id_short = best.id.split("_")[1] if "_" in best.id else best.id[:8]

    # Create cutline for clipping
    cutline_file = output_dir / "cutline.geojson"
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

    downloaded_files = {}

    # Download wanted assets
    print(f"\n📥 Downloading assets: {WANTED_ASSETS}")

    for asset_name in WANTED_ASSETS:
        if asset_name not in assets:
            print(f"   ⚠️ {asset_name}: not available")
            continue

        asset = assets[asset_name]
        asset_url = asset.href
        output_file = (
            output_dir / f"sentinel2_sr_{scene_date}_{scene_id_short}_{asset_name}.tif"
        )

        print(f"   📦 {asset_name}...")

        cmd = [
            "gdalwarp",
            "-overwrite",
            "-cutline",
            str(cutline_file),
            "-crop_to_cutline",
            "-dstalpha",
            "-co",
            "COMPRESS=LZW",
            "-co",
            "TILED=YES",
            f"/vsicurl/{asset_url}",
            str(output_file),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0 and output_file.exists():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"      ✅ Saved: {output_file.name} ({size_mb:.2f} MB)")
            downloaded_files[asset_name] = output_file
        else:
            print(
                f"      ❌ Failed: {result.stderr[:100] if result.stderr else 'unknown error'}"
            )

    # Download multispectral bands (SR data)
    ms_bands = bands if bands else MULTISPECTRAL_BANDS
    print(f"\n📥 Downloading multispectral bands (SR): {ms_bands}")
    for band in ms_bands:
        if band not in assets:
            print(f"   ⚠️ {band}: not available")
            continue

        asset = assets[band]
        asset_url = asset.href
        output_file = (
            output_dir / f"sentinel2_sr_{scene_date}_{scene_id_short}_{band}.tif"
        )

        print(f"   📦 {band}...")

        cmd = [
            "gdalwarp",
            "-overwrite",
            "-cutline",
            str(cutline_file),
            "-crop_to_cutline",
            "-co",
            "COMPRESS=LZW",
            f"/vsicurl/{asset_url}",
            str(output_file),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0 and output_file.exists():
            size_mb = output_file.stat().st_size / (1024 * 1024)
            print(f"      ✅ Saved: {output_file.name} ({size_mb:.2f} MB)")
            downloaded_files[band] = output_file
        else:
            print(f"      ❌ Failed")

    cutline_file.unlink(missing_ok=True)

    if downloaded_files:
        # Create metadata
        meta_file = (
            output_dir / f"sentinel2_sr_{scene_date}_{scene_id_short}_metadata.json"
        )
        metadata = {
            "scene_id": best.id,
            "acquisition_date": best.properties["datetime"],
            "cloud_cover_pct": best.properties.get("eo:cloud_cover"),
            "platform": best.properties.get("platform"),
            "processing_level": "L2A (Surface Reflectance)",
            "source": "AWS Earth Search",
            "downloaded_at": datetime.utcnow().isoformat() + "Z",
            "files": {k: str(v) for k, v in downloaded_files.items()},
        }
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"\n📄 Metadata: {meta_file}")

        # Return visual file as primary if available
        return downloaded_files.get("visual") or list(downloaded_files.values())[0]

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Fetch cloud-free Sentinel-2 Surface Reflectance imagery"
    )
    parser.add_argument(
        "--polygon",
        "-p",
        required=True,
        help="Path to GeoJSON file or JSON string with polygon",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--days", "-d", type=int, default=30, help="Days to look back (default: 30)"
    )
    parser.add_argument(
        "--max-cloud",
        "-c",
        type=float,
        default=10.0,
        help="Maximum cloud cover percentage (default: 10)",
    )
    parser.add_argument(
        "--use-up42",
        action="store_true",
        help="Use UP42 API (requires credentials, checks existing orders first)",
    )
    parser.add_argument(
        "--bands",
        "-b",
        nargs="+",
        default=None,
        help="Bands to download for AWS mode (default: red, green, blue)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Sentinel-2 Surface Reflectance Fetcher")
    print("=" * 60)

    # Load polygon
    try:
        geometry, aoi_data = load_polygon(args.polygon)
        print(f"✅ Loaded polygon: {geometry['type']}")
    except Exception as e:
        print(f"❌ Error loading polygon: {e}")
        sys.exit(1)

    output_dir = Path(args.output)

    if args.use_up42:
        print("\n📡 Using UP42 API (will reuse existing orders)")
        output_file = fetch_via_up42(
            geometry, aoi_data, output_dir, args.days, args.max_cloud
        )
    else:
        print("\n🌍 Using AWS Earth Search (free, public)")
        output_file = fetch_via_aws(
            geometry, output_dir, args.days, args.max_cloud, args.bands
        )

    if output_file:
        print("\n" + "=" * 60)
        print("✅ DONE!")
        print("=" * 60)
        print(f"Output: {output_file}")
    else:
        print("\n❌ Failed to fetch imagery")
        sys.exit(1)


if __name__ == "__main__":
    main()
