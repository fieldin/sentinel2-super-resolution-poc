"""
FastAPI server for serving satellite imagery tiles and metadata.
Includes True Super-Resolution capability using Real-ESRGAN.
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from .settings import get_settings
from .utils import setup_logging, read_json, find_latest_metadata

logger = setup_logging("server")

# Thread pool for SR processing
sr_executor = ThreadPoolExecutor(max_workers=1)

# Store for SR job status
sr_jobs = {}

# Initialize FastAPI app
app = FastAPI(
    title="Sentinel-2 Super-Resolution POC",
    description="Serves satellite imagery tiles with AI Super-Resolution",
    version="2.0.0",
)

# Get settings
settings = get_settings()

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for POC
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
DATA_DIR = Path(settings.data_dir)
TILES_DIR = DATA_DIR / "tiles"
SOURCE_DIR = DATA_DIR / "source"
STATIC_DIR = Path("/app/static")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "up42-sentinel-poc"}


@app.get("/api/config")
async def get_config():
    """
    Get client configuration including Mapbox token.

    Note: In production, consider alternative approaches for token delivery.
    This is acceptable for POC/internal use.
    """
    return {
        "mapboxAccessToken": settings.mapbox_access_token,
        "tileMinZoom": settings.tile_min_zoom,
        "tileMaxZoom": settings.tile_max_zoom,
        "defaultCenter": [-121.487, 36.836],  # Field AOI center
        "defaultZoom": 12,
    }


@app.get("/api/metadata")
async def get_metadata():
    """
    Get tileset and source metadata.

    Returns combined metadata from tileset.json and latest source metadata.
    """
    result = {}

    # Load tileset metadata
    tileset_path = TILES_DIR / "tileset.json"
    if tileset_path.exists():
        result["tileset"] = read_json(tileset_path)
    else:
        result["tileset"] = None
        logger.warning("Tileset metadata not found. Run tile generation first.")

    # Load latest source metadata
    source_meta = find_latest_metadata(SOURCE_DIR)
    result["source"] = source_meta

    # Add availability flag - tiles are in {z}/{x}/{y}.png structure
    result["tilesAvailable"] = (
        tileset_path.exists() and any(TILES_DIR.glob("*/*/*.png"))
        if TILES_DIR.exists()
        else False
    )

    # Check for SR tiles
    sr_tiles_dir = DATA_DIR / "tiles_sr"
    result["srTilesAvailable"] = sr_tiles_dir.exists() and any(
        sr_tiles_dir.glob("*/*/*.png")
    )

    # Check for WOW tiles
    wow_tiles_dir = DATA_DIR / "tiles_wow"
    result["wowTilesAvailable"] = wow_tiles_dir.exists() and any(
        wow_tiles_dir.glob("*/*/*.png")
    )

    # Add tile endpoints
    result["tileEndpoints"] = {
        "original": "/tiles/{z}/{x}/{y}.png",
        "sr": "/tiles_sr/{z}/{x}/{y}.png",
        "wow": "/tiles_wow/{z}/{x}/{y}.png",
    }

    return result


@app.get("/tiles/{z}/{x}/{y}.png")
async def get_tile(z: int, x: int, y: int):
    """
    Serve XYZ raster tiles.

    Args:
        z: Zoom level
        x: Tile X coordinate
        y: Tile Y coordinate

    Returns:
        PNG tile image
    """
    tile_path = TILES_DIR / str(z) / str(x) / f"{y}.png"

    if not tile_path.exists():
        # Return transparent tile for missing tiles
        raise HTTPException(status_code=404, detail="Tile not found")

    return FileResponse(
        tile_path,
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=86400",  # Cache for 1 day
            "Access-Control-Allow-Origin": "*",
        },
    )


# ============================================================
# SUPER-RESOLUTION ENDPOINTS
# ============================================================


class SRRequest(BaseModel):
    """Request body for super-resolution."""

    input_file: Optional[str] = None  # If None, use latest
    scale: int = 4  # 2, 3, or 4
    model: str = "edsr"  # edsr, espcn, or lapsrn


class WowRequest(BaseModel):
    """Request body for WOW super-resolution (Real-ESRGAN x4 + Enhanced)."""

    input_file: Optional[str] = None  # If None, use smart_fetch
    enhance_crops: bool = True  # Apply CLAHE + unsharp mask for crop visibility
    auto_fetch: bool = True  # Automatically fetch best image if needed
    max_age_days: int = 30  # Max age for auto-fetch
    max_cloud_cover: float = 30.0  # Max cloud % for auto-fetch
    force_fetch: bool = False  # Force fetch even if local is good


class PipelineRequest(BaseModel):
    """Request body for full pipeline: fetch → tiles → SR → SR tiles."""

    # Fetch options
    max_age_days: int = 30  # Max age for image search
    max_cloud_cover: float = 30.0  # Max cloud % for image search
    force_fetch: bool = False  # Force fetch even if local is good

    # Tile options
    generate_original_tiles: bool = True  # Generate tiles from original image
    min_zoom: int = 10  # Min zoom level for tiles
    max_zoom: int = 16  # Max zoom level for tiles

    # SR options
    run_sr: bool = True  # Run super-resolution
    sr_type: str = "wow"  # "wow" (Real-ESRGAN x4) or "farm" (EDSR x4)
    enhance_crops: bool = True  # Apply crop enhancement (for WOW)


class SRResponse(BaseModel):
    """Response for SR job."""

    job_id: str
    status: str
    message: str


class PipelineResponse(BaseModel):
    """Response for pipeline job."""

    job_id: str
    status: str
    message: str
    steps: list


def run_sr_job(job_id: str, input_file: Path, scale: int, model: str, output_dir: Path):
    """Background task to run super-resolution."""
    try:
        sr_jobs[job_id]["status"] = "processing"
        sr_jobs[job_id][
            "message"
        ] = f"Applying {model.upper()} x{scale} super-resolution..."

        from .farm_sr import process_farm_sr

        result = process_farm_sr(
            input_tif=input_file,
            output_dir=output_dir,
            scale=scale,
        )

        sr_jobs[job_id]["status"] = "tiling"
        sr_jobs[job_id]["message"] = "Generating tiles from SR image..."

        # Generate tiles from SR output
        sr_tif = result["outputs"].get("sr_tif")
        if sr_tif and Path(sr_tif).exists():
            from .tiling import process_raster_to_tiles

            sr_tiles_dir = DATA_DIR / "tiles_sr"
            process_raster_to_tiles(
                input_path=Path(sr_tif),
                tiles_dir=sr_tiles_dir,
                min_zoom=settings.tile_min_zoom,
                max_zoom=min(settings.tile_max_zoom + 2, 20),  # Higher zoom for SR
            )
            result["tiles_dir"] = str(sr_tiles_dir)

        sr_jobs[job_id]["status"] = "completed"
        sr_jobs[job_id]["message"] = "Super-resolution complete!"
        sr_jobs[job_id]["result"] = result

    except Exception as e:
        logger.error(f"SR job {job_id} failed: {e}")
        sr_jobs[job_id]["status"] = "failed"
        sr_jobs[job_id]["message"] = str(e)


def run_wow_job(
    job_id: str,
    input_file: Optional[Path],
    output_dir: Path,
    enhance_crops: bool,
    auto_fetch: bool = True,
    max_age_days: int = 30,
    max_cloud_cover: float = 30.0,
    force_fetch: bool = False,
):
    """Background task to run WOW super-resolution (Real-ESRGAN x4 + Enhanced)."""
    try:
        # Step 1: Smart fetch if needed
        if input_file is None and auto_fetch:
            sr_jobs[job_id]["status"] = "fetching"
            sr_jobs[job_id][
                "message"
            ] = f"🔍 Finding best image (last {max_age_days} days, cloud ≤{max_cloud_cover}%)..."

            from .smart_fetch import ensure_best_image

            input_file, fetch_metadata = ensure_best_image(
                source_dir=SOURCE_DIR,
                max_age_days=max_age_days,
                max_cloud_cover=max_cloud_cover,
                force_fetch=force_fetch,
            )

            sr_jobs[job_id]["input_file"] = str(input_file)
            sr_jobs[job_id]["fetch_metadata"] = fetch_metadata
            sr_jobs[job_id][
                "message"
            ] = f"✅ Using: {input_file.name} (cloud: {fetch_metadata.get('cloud_cover_pct', 'N/A')}%)"

        # Step 2: Run WOW SR
        sr_jobs[job_id]["status"] = "processing"
        sr_jobs[job_id]["message"] = "Stage 1/2: Real-ESRGAN x4 (GAN upscaling)..."

        from .wow_sr import process_wow_sr

        result = process_wow_sr(
            input_tif=input_file,
            output_dir=output_dir,
            enhance_crops=enhance_crops,
        )

        sr_jobs[job_id]["status"] = "tiling"
        sr_jobs[job_id]["message"] = "Generating tiles from WOW SR image..."

        # Generate tiles from SR output
        sr_tif = result["outputs"].get("sr_tif")
        if sr_tif and Path(sr_tif).exists():
            from .tiling import process_raster_to_tiles

            wow_tiles_dir = DATA_DIR / "tiles_wow"
            process_raster_to_tiles(
                input_path=Path(sr_tif),
                tiles_dir=wow_tiles_dir,
                min_zoom=settings.tile_min_zoom,
                max_zoom=min(settings.tile_max_zoom + 2, 20),  # Higher zoom for SR
            )
            result["tiles_dir"] = str(wow_tiles_dir)

        sr_jobs[job_id]["status"] = "completed"
        sr_jobs[job_id]["message"] = "WOW Super-resolution complete! 🌟"
        sr_jobs[job_id]["result"] = result

    except Exception as e:
        logger.error(f"WOW job {job_id} failed: {e}")
        sr_jobs[job_id]["status"] = "failed"
        sr_jobs[job_id]["message"] = str(e)


@app.post("/api/sr", response_model=SRResponse)
async def start_super_resolution(request: SRRequest, background_tasks: BackgroundTasks):
    """
    Start a super-resolution job on Sentinel-2 imagery.

    - If input_file is not specified, uses the latest downloaded GeoTIFF
    - scale: 2 (5m effective) or 4 (2.5m effective)

    Returns job_id to track progress.
    """
    # Find input file
    if request.input_file:
        input_file = Path(request.input_file)
    else:
        # Find latest GeoTIFF in source dir
        tif_files = sorted(
            SOURCE_DIR.glob("*.tif"), key=lambda x: x.stat().st_mtime, reverse=True
        )
        if not tif_files:
            raise HTTPException(
                status_code=404, detail="No GeoTIFF files found. Run fetch first."
            )
        input_file = tif_files[0]

    if not input_file.exists():
        raise HTTPException(
            status_code=404, detail=f"Input file not found: {input_file}"
        )

    # Validate scale
    if request.scale not in [2, 3, 4]:
        raise HTTPException(status_code=400, detail="Scale must be 2, 3, or 4")

    # Validate model
    if request.model not in ["edsr", "espcn", "lapsrn"]:
        raise HTTPException(
            status_code=400, detail="Model must be edsr, espcn, or lapsrn"
        )

    # Create job
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = DATA_DIR / "sr" / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    sr_jobs[job_id] = {
        "status": "queued",
        "message": "Job queued",
        "input_file": str(input_file),
        "scale": request.scale,
        "model": request.model,
        "output_dir": str(output_dir),
        "created_at": datetime.now().isoformat(),
    }

    # Run in background
    background_tasks.add_task(
        run_sr_job, job_id, input_file, request.scale, request.model, output_dir
    )

    return SRResponse(
        job_id=job_id,
        status="queued",
        message=f"SR job started: {input_file.name} → x{request.scale}",
    )


@app.get("/api/sr/{job_id}")
async def get_sr_status(job_id: str):
    """Get status of a super-resolution job."""
    if job_id not in sr_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return sr_jobs[job_id]


@app.get("/api/sr")
async def list_sr_jobs():
    """List all super-resolution jobs."""
    return {"jobs": sr_jobs}


# ============================================================
# WOW SUPER-RESOLUTION ENDPOINT (Real-ESRGAN x4 + Enhanced)
# ============================================================


@app.post("/api/wow", response_model=SRResponse)
async def start_wow_sr(request: WowRequest, background_tasks: BackgroundTasks):
    """
    Start WOW super-resolution: Real-ESRGAN x4 + Enhanced Post-Processing.

    This approach provides superior quality at z18 for agricultural imagery:
    - Real-ESRGAN x4: High-quality GAN-based upscaling
    - Enhanced post-processing: CLAHE + unsharp mask + vegetation boost

    Features:
    - auto_fetch: Automatically fetches best image (latest + clearest) from last 30 days
    - Only fetches from UP42/AWS if local is outdated or missing
    - max_age_days: How far back to look (default 30)
    - max_cloud_cover: Maximum cloud % (default 30)

    Total scale: x4 (10m → 2.5m effective resolution)

    Returns job_id to track progress.
    """
    # Determine input file
    input_file = None

    if request.input_file:
        # Explicit file specified
        input_file = Path(request.input_file)
        if not input_file.exists():
            raise HTTPException(
                status_code=404, detail=f"Input file not found: {input_file}"
            )
    elif not request.auto_fetch:
        # No auto-fetch, find latest local file
        tif_files = sorted(
            SOURCE_DIR.glob("*.tif"), key=lambda x: x.stat().st_mtime, reverse=True
        )
        if not tif_files:
            raise HTTPException(
                status_code=404,
                detail="No GeoTIFF files found. Enable auto_fetch=true or run fetch first.",
            )
        input_file = tif_files[0]
    # If auto_fetch=True and no input_file, let run_wow_job handle it with smart_fetch

    # Create job
    job_id = f"wow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = DATA_DIR / "wow" / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    sr_jobs[job_id] = {
        "status": "queued",
        "message": "WOW job queued (Real-ESRGAN x4 + Enhanced)",
        "input_file": str(input_file) if input_file else "auto_fetch",
        "pipeline": "RealESRGAN_x4 + Enhanced",
        "scale": 4,
        "enhance_crops": request.enhance_crops,
        "auto_fetch": request.auto_fetch,
        "max_age_days": request.max_age_days,
        "max_cloud_cover": request.max_cloud_cover,
        "output_dir": str(output_dir),
        "created_at": datetime.now().isoformat(),
    }

    # Run in background
    background_tasks.add_task(
        run_wow_job,
        job_id,
        input_file,
        output_dir,
        request.enhance_crops,
        request.auto_fetch,
        request.max_age_days,
        request.max_cloud_cover,
        request.force_fetch,
    )

    # Build response message
    if input_file:
        msg = f"🌟 WOW SR started: {input_file.name} → Real-ESRGAN x4 + Enhanced"
    else:
        msg = f"🌟 WOW SR started: auto-fetching best image (last {request.max_age_days}d, cloud ≤{request.max_cloud_cover}%)"

    return SRResponse(
        job_id=job_id,
        status="queued",
        message=msg,
    )


@app.get("/tiles_wow/{z}/{x}/{y}.png")
async def get_wow_tile(z: int, x: int, y: int):
    """Serve WOW super-resolution XYZ tiles."""
    wow_tiles_dir = DATA_DIR / "tiles_wow"
    tile_path = wow_tiles_dir / str(z) / str(x) / f"{y}.png"

    if not tile_path.exists():
        raise HTTPException(status_code=404, detail="WOW tile not found")

    return FileResponse(
        tile_path,
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=86400",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ============================================================
# FULL PIPELINE ENDPOINT (Fetch → Tiles → SR → SR Tiles)
# ============================================================


def run_pipeline_job(
    job_id: str,
    max_age_days: int,
    max_cloud_cover: float,
    force_fetch: bool,
    generate_original_tiles: bool,
    min_zoom: int,
    max_zoom: int,
    run_sr: bool,
    sr_type: str,
    enhance_crops: bool,
):
    """
    Background task to run the full pipeline:
    1. Smart fetch (get best Sentinel-2 image)
    2. Generate original tiles
    3. Run super-resolution
    4. Generate SR tiles
    """
    try:
        steps_completed = []

        # ============================================================
        # STEP 1: Smart Fetch
        # ============================================================
        sr_jobs[job_id]["status"] = "fetching"
        sr_jobs[job_id]["current_step"] = 1
        sr_jobs[job_id][
            "message"
        ] = f"📡 Step 1/4: Finding best image (last {max_age_days} days, cloud ≤{max_cloud_cover}%)..."

        from .smart_fetch import ensure_best_image

        input_file, fetch_metadata = ensure_best_image(
            source_dir=SOURCE_DIR,
            max_age_days=max_age_days,
            max_cloud_cover=max_cloud_cover,
            force_fetch=force_fetch,
        )

        sr_jobs[job_id]["input_file"] = str(input_file)
        sr_jobs[job_id]["fetch_metadata"] = fetch_metadata
        steps_completed.append(
            {
                "step": 1,
                "name": "fetch",
                "status": "completed",
                "message": f"✅ Image: {input_file.name}",
                "details": {
                    "file": str(input_file),
                    "cloud_cover": fetch_metadata.get("cloud_cover_pct"),
                    "acquisition_date": fetch_metadata.get("acquisition_date"),
                },
            }
        )

        # ============================================================
        # STEP 2: Generate Original Tiles
        # ============================================================
        if generate_original_tiles:
            sr_jobs[job_id]["status"] = "tiling_original"
            sr_jobs[job_id]["current_step"] = 2
            sr_jobs[job_id]["message"] = "🗺️ Step 2/4: Generating original tiles..."

            from .tiling import process_raster_to_tiles

            tiles_metadata = process_raster_to_tiles(
                input_path=input_file,
                tiles_dir=TILES_DIR,
                min_zoom=min_zoom,
                max_zoom=max_zoom,
            )

            steps_completed.append(
                {
                    "step": 2,
                    "name": "original_tiles",
                    "status": "completed",
                    "message": f"✅ Tiles generated (z{min_zoom}-{max_zoom})",
                    "details": tiles_metadata,
                }
            )
        else:
            steps_completed.append(
                {
                    "step": 2,
                    "name": "original_tiles",
                    "status": "skipped",
                    "message": "⏭️ Skipped (generate_original_tiles=false)",
                }
            )

        # ============================================================
        # STEP 3: Super-Resolution
        # ============================================================
        sr_output = None
        if run_sr:
            sr_jobs[job_id]["status"] = "processing_sr"
            sr_jobs[job_id]["current_step"] = 3
            sr_jobs[job_id][
                "message"
            ] = f"🚀 Step 3/4: Running {sr_type.upper()} super-resolution..."

            output_dir = DATA_DIR / "sr" / job_id
            output_dir.mkdir(parents=True, exist_ok=True)

            if sr_type == "wow":
                from .wow_sr import process_wow_sr

                sr_result = process_wow_sr(
                    input_tif=input_file,
                    output_dir=output_dir,
                    enhance_crops=enhance_crops,
                )
            else:  # farm
                from .farm_sr import process_farm_sr

                sr_result = process_farm_sr(
                    input_tif=input_file,
                    output_dir=output_dir,
                    scale=4,
                )

            sr_output = sr_result["outputs"].get("sr_tif")
            steps_completed.append(
                {
                    "step": 3,
                    "name": "super_resolution",
                    "status": "completed",
                    "message": f"✅ SR complete ({sr_type.upper()} x4)",
                    "details": {
                        "output": sr_output,
                        "scale": 4,
                        "type": sr_type,
                    },
                }
            )
        else:
            steps_completed.append(
                {
                    "step": 3,
                    "name": "super_resolution",
                    "status": "skipped",
                    "message": "⏭️ Skipped (run_sr=false)",
                }
            )

        # ============================================================
        # STEP 4: Generate SR Tiles
        # ============================================================
        if run_sr and sr_output and Path(sr_output).exists():
            sr_jobs[job_id]["status"] = "tiling_sr"
            sr_jobs[job_id]["current_step"] = 4
            sr_jobs[job_id]["message"] = "🗺️ Step 4/4: Generating SR tiles..."

            from .tiling import process_raster_to_tiles

            # Use appropriate tiles directory based on SR type
            if sr_type == "wow":
                sr_tiles_dir = DATA_DIR / "tiles_wow"
            else:
                sr_tiles_dir = DATA_DIR / "tiles_sr"

            sr_tiles_metadata = process_raster_to_tiles(
                input_path=Path(sr_output),
                tiles_dir=sr_tiles_dir,
                min_zoom=min_zoom,
                max_zoom=min(max_zoom + 2, 20),  # Higher zoom for SR
            )

            steps_completed.append(
                {
                    "step": 4,
                    "name": "sr_tiles",
                    "status": "completed",
                    "message": f"✅ SR tiles generated (z{min_zoom}-{min(max_zoom + 2, 20)})",
                    "details": sr_tiles_metadata,
                }
            )
        elif run_sr:
            steps_completed.append(
                {
                    "step": 4,
                    "name": "sr_tiles",
                    "status": "failed",
                    "message": "❌ SR output not found",
                }
            )
        else:
            steps_completed.append(
                {
                    "step": 4,
                    "name": "sr_tiles",
                    "status": "skipped",
                    "message": "⏭️ Skipped (SR not run)",
                }
            )

        # ============================================================
        # COMPLETE
        # ============================================================
        sr_jobs[job_id]["status"] = "completed"
        sr_jobs[job_id]["current_step"] = 4
        sr_jobs[job_id]["message"] = "🎉 Pipeline complete!"
        sr_jobs[job_id]["steps"] = steps_completed
        sr_jobs[job_id]["completed_at"] = datetime.now().isoformat()

    except Exception as e:
        logger.error(f"Pipeline job {job_id} failed: {e}")
        sr_jobs[job_id]["status"] = "failed"
        sr_jobs[job_id]["message"] = f"❌ Failed: {str(e)}"
        sr_jobs[job_id]["error"] = str(e)


@app.post("/api/pipeline", response_model=PipelineResponse)
async def start_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    """
    🚀 Run the FULL pipeline: Fetch → Tiles → SR → SR Tiles

    This endpoint orchestrates the complete workflow:

    **Step 1: Smart Fetch**
    - Checks local images vs remote (AWS Earth Search)
    - Downloads if local is outdated or missing
    - Filters by age (max_age_days) and cloud cover (max_cloud_cover)

    **Step 2: Generate Original Tiles**
    - Creates XYZ tiles from source image
    - Zoom levels: min_zoom to max_zoom

    **Step 3: Super-Resolution**
    - "wow": Real-ESRGAN x4 + crop enhancement (recommended)
    - "farm": EDSR x4 (faster, lower quality)

    **Step 4: Generate SR Tiles**
    - Creates XYZ tiles from SR image
    - Higher zoom levels for enhanced detail

    Returns job_id to track progress via GET /api/pipeline/{job_id}
    """
    # Create job
    job_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    sr_jobs[job_id] = {
        "status": "queued",
        "current_step": 0,
        "message": "🚀 Pipeline queued...",
        "config": {
            "max_age_days": request.max_age_days,
            "max_cloud_cover": request.max_cloud_cover,
            "force_fetch": request.force_fetch,
            "generate_original_tiles": request.generate_original_tiles,
            "min_zoom": request.min_zoom,
            "max_zoom": request.max_zoom,
            "run_sr": request.run_sr,
            "sr_type": request.sr_type,
            "enhance_crops": request.enhance_crops,
        },
        "steps": [],
        "created_at": datetime.now().isoformat(),
    }

    # Run in background
    background_tasks.add_task(
        run_pipeline_job,
        job_id,
        request.max_age_days,
        request.max_cloud_cover,
        request.force_fetch,
        request.generate_original_tiles,
        request.min_zoom,
        request.max_zoom,
        request.run_sr,
        request.sr_type,
        request.enhance_crops,
    )

    return PipelineResponse(
        job_id=job_id,
        status="queued",
        message=f"🚀 Pipeline started: Fetch → Tiles → {request.sr_type.upper()} SR → SR Tiles",
        steps=["fetch", "original_tiles", "super_resolution", "sr_tiles"],
    )


@app.get("/api/pipeline/{job_id}")
async def get_pipeline_status(job_id: str):
    """
    Get status of a pipeline job.

    Returns detailed progress including:
    - Current step (1-4)
    - Status of each completed step
    - Overall status
    """
    if job_id not in sr_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return sr_jobs[job_id]


@app.get("/api/pipelines")
async def list_pipelines():
    """List all pipeline jobs (past and current)."""
    pipeline_jobs = {k: v for k, v in sr_jobs.items() if k.startswith("pipeline_")}
    return {
        "jobs": pipeline_jobs,
        "count": len(pipeline_jobs),
    }


@app.get("/tiles_sr/{z}/{x}/{y}.png")
async def get_sr_tile(z: int, x: int, y: int):
    """Serve super-resolution XYZ tiles."""
    sr_tiles_dir = DATA_DIR / "tiles_sr"
    tile_path = sr_tiles_dir / str(z) / str(x) / f"{y}.png"

    if not tile_path.exists():
        raise HTTPException(status_code=404, detail="SR tile not found")

    return FileResponse(
        tile_path,
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=86400",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/api/sr/download/{job_id}/{filename}")
async def download_sr_file(job_id: str, filename: str):
    """Download a super-resolution output file."""
    if job_id not in sr_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    output_dir = Path(sr_jobs[job_id]["output_dir"])
    file_path = output_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path)


# Mount static files for Angular app (if exists)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# SPA fallback - serve index.html for all non-API routes
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    """
    Serve Angular SPA.

    All non-API routes fall back to index.html for client-side routing.
    """
    # Check for static files first
    static_path = STATIC_DIR / full_path
    if static_path.exists() and static_path.is_file():
        return FileResponse(static_path)

    # Check for index.html at root or in static
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)

    # Fallback - serve a simple HTML page if Angular not built
    return JSONResponse(
        status_code=200,
        content={
            "message": "UP42 Sentinel-2 POC Server",
            "endpoints": {
                "health": "/health",
                "config": "/api/config",
                "metadata": "/api/metadata",
                "tiles": "/tiles/{z}/{x}/{y}.png",
            },
            "note": "Build Angular client and place in /app/static for web UI",
        },
    )


def start_server():
    """Start the Uvicorn server."""
    logger.info(f"Starting server on {settings.server_host}:{settings.server_port}")
    uvicorn.run(
        "app.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    start_server()
