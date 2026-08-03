# Pipeline and API deep dive

FastAPI app in `server/app/main.py`. Swagger at `/docs` when server running.

## Job storage

All async jobs share one dict: `sr_jobs[job_id]`.

| Prefix | Endpoint | Background fn |
|--------|----------|---------------|
| `pipeline_*` | `/api/pipeline` | `run_pipeline_job` |
| `wow_*` | `/api/wow`, `/api/enhance` | `run_wow_job` |
| `vectors_*` | `/api/vectors` | `run_vector_extraction_job` |
| timestamp | `/api/sr` | `run_sr_job` |

**Limitation:** Jobs lost on process restart. SR uses `ThreadPoolExecutor(max_workers=1)` — only one heavy SR at a time; enhance upload has queue (`pending_enhance_queue`).

## Core endpoints

### Health & config

| Route | Returns |
|-------|---------|
| `GET /health` | `{"status":"healthy"}` |
| `GET /api/config` | Mapbox token, zoom defaults, center `[-121.487, 36.836]` |
| `GET /api/metadata` | Tileset, source meta, availability flags, endpoints |

`metadata` response keys used by client:

- `tilesAvailable`, `wowTilesAvailable`, `srTilesAvailable`
- `tileset`, `wowTileset` (bounds, minzoom, maxzoom, tileTemplate)
- `source` (scene_id, acquisition_date, cloud_cover_pct, bbox)
- `tileEndpoints`: `original`, `sr`, `wow`
- `vectorsAvailable`, `vectorEndpoints`

### Tiles (PNG)

| Route | Directory |
|-------|-----------|
| `/tiles/{z}/{x}/{y}.png` | `data/tiles/` |
| `/tiles_sr/{z}/{x}/{y}.png` | `data/tiles_sr/` |
| `/tiles_wow/{z}/{x}/{y}.png` | `data/tiles_wow/` |

404 if missing. Headers: `Cache-Control: no-cache`, CORS `*`.

### Full pipeline

```http
POST /api/pipeline
Content-Type: application/json

{
  "max_age_days": 30,
  "max_cloud_cover": 30.0,
  "force_fetch": false,
  "generate_original_tiles": true,
  "min_zoom": 10,
  "max_zoom": 16,
  "run_sr": true,
  "sr_type": "wow",
  "enhance_crops": true
}
```

Response:

```json
{
  "job_id": "pipeline_20260428_104640",
  "status": "queued",
  "message": "🚀 Pipeline started: Fetch → Tiles → WOW SR → SR Tiles",
  "steps": ["fetch", "original_tiles", "super_resolution", "sr_tiles"]
}
```

Poll: `GET /api/pipeline/{job_id}`

Status progression: `queued` → `fetching` → `tiling_original` → `processing_sr` → `tiling_sr` → `completed` | `failed`

Completed job includes `steps[]` with per-step status and `fetch_metadata`.

List: `GET /api/pipelines`

### WOW SR only

```http
POST /api/wow
{
  "auto_fetch": true,
  "max_age_days": 30,
  "max_cloud_cover": 30.0,
  "enhance_crops": true,
  "force_fetch": false
}
```

### Image upload enhance

```http
POST /api/enhance
multipart: image=..., model=realesrgan_x4|realesrgan_anime
```

Max upload: `MAX_UPLOAD_BYTES` (default 50 MB). Poll `GET /api/sr/{job_id}`.

### Standard SR (EDSR)

```http
POST /api/sr
{"scale": 4, "model": "edsr", "input_file": null}
```

### Vectors

```http
POST /api/vectors
{
  "ndvi_threshold": 0.3,
  "min_area_ha": 0.1,
  "max_area_ha": 500.0,
  "simplify_tolerance_m": 5.0
}
```

Serve: `GET /vectors/fields.geojson`, `GET /vectors/zones.geojson`

Meta: `GET /api/vectors/metadata`

## smart_fetch internals

`ensure_best_image(source_dir, max_age_days, max_cloud_cover, force_fetch)`:

1. `get_local_images` — parse `*_meta.json` for cloud % and acquisition date.
2. `select_best_local_image` — newest with cloud under threshold.
3. Remote search (STAC) — compare cloud + date vs local.
4. Download to `data/source/` via `fetch` / `up42_client` if needed.
5. Return `(Path to tif, metadata dict)`.

CLI: `python -m app.smart_fetch` or `make smart-fetch`.

## WOW SR internals

`process_wow_sr(input_tif, output_dir, enhance_crops, model)`:

1. Read GeoTIFF bands 1–3 as RGB (`rasterio`).
2. `RealESRGAN.enhance` — tiled inference (`tile_size=256`).
3. Optional `enhance_crops` post-pass in `wow_sr.py`.
4. Write `*_sr.tif` + metadata JSON under `output_dir`.
5. Pipeline then calls `process_raster_to_tiles` on SR output.

## Tiling

`tiling.process_raster_to_tiles(input_path, tiles_dir, min_zoom, max_zoom, tile_template?)`:

- Writes `{z}/{x}/{y}.png` (256 px).
- Writes/updates `tileset.json` (bounds, minzoom, maxzoom, tileTemplate).
- WOW pipeline uses `tile_template="/tiles_wow/{z}/{x}/{y}.png"`.
- SR tiles may use `max_zoom + 2` capped at 20.

## Vector extraction v2 (CLI)

```bash
python -m app.vector_extraction_v2 \
  --aoi config/aoi.geojson \
  --rasters data/source/*.tif \
  --out data/vectors
```

Flags: `--no-osm`, `--no-zones`, NDVI thresholds, etc.

`make vectors-v2`, `make multiband`, `make vectors-ndvi`.

## Environment variables

From `server/app/settings.py` / `.env.example`:

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `MAPBOX_ACCESS_TOKEN` | **Yes** | — | Client map |
| `UP42_USERNAME` | No | "" | UP42 fetch |
| `UP42_PASSWORD` | No | "" | UP42 fetch |
| `UP42_PROJECT_ID` | No | "" | UP42 project |
| `DAYS_LOOKBACK` | No | 30 | Search window |
| `MAX_CLOUD_PCT` | No | 10 | Settings default (pipeline API uses request body) |
| `TILE_MIN_ZOOM` | No | 10 | Tile gen |
| `TILE_MAX_ZOOM` | No | 16 | Tile gen |
| `DATA_DIR` | No | `/app/data` | Data root |
| `SERVER_PORT` | No | 8080 | Listen port |

## Python module map

| Module | Role |
|--------|------|
| `main.py` | FastAPI app, routes, job runners |
| `smart_fetch.py` | Best scene selection + download |
| `fetch.py` / `fetch_multiband.py` | Manual / multiband fetch |
| `up42_client.py` | UP42 STAC + download |
| `wow_sr.py` | WOW pipeline wrapper |
| `cnn_super_resolution.py` | Real-ESRGAN wrapper |
| `farm_sr.py` | EDSR / lighter SR |
| `super_resolution.py` | Legacy SR helpers |
| `tiling.py` | GeoTIFF → XYZ |
| `tile.py` | CLI tile generation |
| `vector_extraction.py` | v1 polygons |
| `vector_extraction_v2.py` | v2 gradient + zones |
| `generate_vectors.py` | v1 CLI |
| `esrgan_tiles.py` | ESRGAN tile experiments |
| `swinir.py` | SwinIR experiments (Makefile references legacy) |
| `sr_cli.py` | CLI for farm SR |
| `utils.py` | JSON IO, logging |

## Testing pipeline locally

```bash
make up
curl http://localhost:8080/health
make pipeline
make pipeline-watch JOB=pipeline_YYYYMMDD_HHMMSS
curl http://localhost:8080/api/metadata | jq .
```

SR step is CPU/GPU heavy — expect several minutes on CPU-only Docker.

## Common failures

| Symptom | Cause |
|---------|--------|
| `No GeoTIFF files found` | Fetch failed or empty `data/source/` |
| Pipeline `failed` at fetch | UP42 creds / network / AOI outside coverage |
| WOW OOM | Large AOI — reduce AOI or increase container memory |
| 404 tiles | Pipeline not finished or wrong z/x/y |
| Empty metadata | Run pipeline or copy prebuilt `data/` |
| Jobs 404 after restart | `sr_jobs` in-memory only |
