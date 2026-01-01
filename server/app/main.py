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

    # Add tile endpoints
    result["tileEndpoints"] = {
        "original": "/tiles/{z}/{x}/{y}.png",
        "sr": "/tiles_sr/{z}/{x}/{y}.png",
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


class SRResponse(BaseModel):
    """Response for SR job."""

    job_id: str
    status: str
    message: str


def run_sr_job(job_id: str, input_file: Path, scale: int, model: str, output_dir: Path):
    """Background task to run super-resolution."""
    try:
        sr_jobs[job_id]["status"] = "processing"
        sr_jobs[job_id][
            "message"
        ] = f"Applying {model.upper()} x{scale} super-resolution..."

        from .super_resolution import process_sentinel2_sr

        result = process_sentinel2_sr(
            input_tif=input_file,
            output_dir=output_dir,
            scale=scale,
            model_type=model,
        )

        sr_jobs[job_id]["status"] = "tiling"
        sr_jobs[job_id]["message"] = "Generating tiles from SR image..."

        # Generate tiles from SR output
        sr_tif = result["outputs"].get("sr_tif")
        if sr_tif and Path(sr_tif).exists():
            from .tiling import generate_tiles

            sr_tiles_dir = DATA_DIR / "tiles_sr"
            generate_tiles(
                input_file=Path(sr_tif),
                output_dir=sr_tiles_dir,
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
