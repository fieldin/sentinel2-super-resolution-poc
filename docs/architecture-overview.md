# Architecture overview

## Problem

Sentinel-2 L2A imagery is **10 m** native resolution — insufficient for high-zoom field detail (crop rows, boundaries at z18+). This POC tests:

1. **Server-side super-resolution** (Real-ESRGAN x4 → ~2.5 m effective).
2. **XYZ tile serving** for web maps (original + SR layers).
3. **Vector field boundaries** (NDVI / gradient segmentation) for crisp overlays at extreme zoom.
4. **fld-map integration** — polygon draw/edit on SR basemap (Skycuse-like UX experiments).

## System diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  Angular client (client/)                                        │
│  • map.component — fld-map + Leaflet SR tile overlay             │
│  • enhance.component — upload image → Real-ESRGAN API            │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP (same origin :8080)
┌───────────────────────────▼─────────────────────────────────────┐
│  FastAPI (server/app/main.py)                                    │
│  POST /api/pipeline  │  GET /tiles*, /api/metadata, /vectors/*  │
└───────────┬───────────────────────────────┬─────────────────────┘
            │                               │
     smart_fetch.py                    wow_sr.py + tiling.py
     (AWS STAC / UP42)                 (Real-ESRGAN + CLAHE)
            │                               │
            ▼                               ▼
┌───────────────────┐              ┌────────────────────────────┐
│ data/source/*.tif │              │ data/tiles/ tiles_wow/     │
│ + *_meta.json     │              │ data/vectors/fields.geojson│
└───────────────────┘              └────────────────────────────┘
```

Single Docker image: Python server + built Angular static files in `/app/static`.

## Pipeline (4 steps)

`POST /api/pipeline` runs in a **background task** (`run_pipeline_job`):

| Step | Name | What happens |
|------|------|----------------|
| 1 | **fetch** | `smart_fetch.ensure_best_image` — pick latest + lowest cloud in window; download GeoTIFF if remote wins |
| 2 | **original_tiles** | `tiling.process_raster_to_tiles` → `data/tiles/{z}/{x}/{y}.png` + `tileset.json` |
| 3 | **super_resolution** | `wow` → `process_wow_sr` (Real-ESRGAN x4 + crop enhancement) **or** `farm` → EDSR x4 |
| 4 | **sr_tiles** | Tiles from SR GeoTIFF → `data/tiles_wow/` (wow) or `data/tiles_sr/` (farm) |

Job state lives in in-memory dict `sr_jobs` (not persisted across restarts).

## WOW super-resolution stack

`wow_sr.py` + `cnn_super_resolution.RealESRGAN`:

1. **Real-ESRGAN x4** — GAN upscale (10 m → 2.5 m effective).
2. **Post-processing** (when `enhance_crops=true`):
   - CLAHE (local contrast)
   - Unsharp mask
   - Vegetation / green channel boost

Models exposed via `/api/enhance` upload:

| Model id | Use |
|----------|-----|
| `realesrgan_x4` | General imagery |
| `realesrgan_anime` | Sharp edges, text/plates |

## Smart fetch strategy

`smart_fetch.py`:

1. Scan `data/source/` for local `.tif` + sidecar `*_meta.json`.
2. Filter by `max_age_days` and `max_cloud_cover`.
3. Query remote catalog (AWS Earth Search STAC primary; UP42 fallback if configured).
4. Download only if remote scene beats local (or `force_fetch`).

Default POC AOI: `config/aoi.geojson` (Fieldin test area ~ Salinas / JV Farms region).

## Vector intelligence

Two extraction paths:

| Path | Module | Features |
|------|--------|----------|
| v1 API | `vector_extraction.py` | NDVI or HSV mask → polygons |
| v2 CLI | `vector_extraction_v2.py` | Gradient watershed, optional OSM roads, management zones |

Outputs:

- `data/vectors/fields.geojson` — served at `GET /vectors/fields.geojson`
- `data/vectors/zones.geojson` — management zones (v2 with `--zones`)

`make vectors-ndvi` — fetch real B04+B08 multiband → compute NDVI → v2 extraction.

## Data directory layout

```
data/
├── source/           # Raw Sentinel-2 GeoTIFF + *_meta.json
├── tiles/            # Original XYZ + tileset.json
├── tiles_sr/         # Farm/EDSR SR tiles
├── tiles_wow/        # WOW (Real-ESRGAN) tiles + tileset.json
├── sr/               # Per-job SR output dirs
├── wow/              # Per-job WOW output dirs
├── vectors/          # fields.geojson, zones.geojson, extraction_metadata.json
└── uploads/          # User uploads for /api/enhance
```

**Important:** `data/` is bind-mounted in Docker / should be backed up for staging — pod restart without persistent volume loses tiles unless baked into image.

## Resolution & zoom

| Layer | Native | After WOW x4 | Tile zoom (typical) |
|-------|--------|--------------|---------------------|
| Sentinel-2 | 10 m | ~2.5 m effective | z10–16 built, client overzoom to z22 |

Client uses `maxNativeZoom` from `tileset.json` and allows overzoom in Leaflet.

## Data sources

| Source | Auth | Notes |
|--------|------|-------|
| **AWS Earth Search STAC** | Public | Primary catalog in smart fetch |
| **UP42** | `UP42_USERNAME` / `UP42_PASSWORD` | Optional fallback |
| **Pre-loaded data** | — | POC can run with existing `data/` only |

## Relationship to Fieldin products

This is a **standalone POC** — not wired into fld-app-new production map. Experiments inform:

- High-zoom basemap quality for field maps
- Vector boundary extraction for field digitization
- fld-map polygon edit UX on satellite background

Potential integration paths: tile URL as custom layer in work-map, or vector GeoJSON into block/field APIs.

## Key files (read order)

1. `server/app/main.py` — all HTTP routes + pipeline orchestration
2. `server/app/smart_fetch.py` — imagery selection
3. `server/app/wow_sr.py` — SR pipeline
4. `server/app/tiling.py` — raster → XYZ
5. `client/src/app/map/map.component.ts` — map viewer
6. `Makefile` — operational commands
