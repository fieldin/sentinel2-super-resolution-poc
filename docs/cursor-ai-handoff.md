# Cursor AI handoff — sentinel2-super-resolution-poc

## Mandatory reading order

1. [docs/README.md](./README.md)
2. [architecture-overview.md](./architecture-overview.md)
3. [pipeline-and-api-deep-dive.md](./pipeline-and-api-deep-dive.md)
4. [client-and-deployment.md](./client-and-deployment.md)

## Project constraints

| Rule | Detail |
|------|--------|
| Branch names | No `/` in git branches |
| Commits | Only when user explicitly asks |
| Secrets | Do not commit `.env`; Mapbox/UP42 in K8s yaml are POC debt |
| POC scope | Standalone — not fld-app-new production |

## Dev quick start

```bash
cd sentinel2-super-resolution-poc
cp .env.example .env   # MAPBOX_ACCESS_TOKEN required
make up
make pipeline
open http://localhost:8080
```

## Task playbooks

### Regenerate tiles for staging

```bash
make pipeline
# or on staging pod — curl POST /api/pipeline (long running)
./deploy.sh latest --skip-build   # if only server code changed
```

### No tiles in UI

1. `curl localhost:8080/api/metadata | jq .tilesAvailable, .wowTilesAvailable`
2. If false → `make pipeline` or restore `data/` volume.
3. Check browser network for 404 on `/tiles_wow/...`.

### Change AOI

Edit `config/aoi.geojson`, then `force_fetch: true` on pipeline or `make clean` + pipeline.

### Run vectors after SR

```bash
make pipeline-full
# or: make vectors-v2 / make vectors-ndvi
```

Toggle field boundaries in UI (if wired) or open `/vectors/fields.geojson`.

### Debug SR failure

- Container logs: `make logs`
- Job status: `make pipeline-status JOB=pipeline_...`
- Memory: WOW needs ~1–2 GiB; increase Docker/K8s limits.
- CPU-only Real-ESRGAN is slow but should complete for small AOIs.

### Change tile zoom range

- Pipeline body: `min_zoom`, `max_zoom`
- Settings: `TILE_MIN_ZOOM`, `TILE_MAX_ZOOM`
- Client reads `tileset.minzoom` / `maxzoom` from metadata.

### fld-map polygon issues

See `map.component.ts` — edit/draw/cancel hooks on `deckLayersService` and `mapControlsService`. Storybook parity comments in file.

### Deploy to staging

```bash
./deploy.sh latest
curl http://sentinel-poc-stg.fieldintech.com/health
```

Requires: Docker, AWS ECR login (`awsnew` profile), kubectl access.

## Key grep

```bash
rg "sr_jobs|run_pipeline|process_wow" server/app
rg "FldMapModule|addSRTileLayer" client/src
rg "sentinel-poc" k8s deploy.sh
```

## Do not

- Assume jobs survive restart (`sr_jobs` is in-memory).
- Run `make clean` on staging without backup if tiles are only local.
- Merge Mapbox keys from map.component into shared fld-app-new env without review.

## Repo name confusion

- Git folder: `sentinel2-super-resolution-poc`
- Docker image: `sentinel-poc`
- Old path in `gideline.txt`: `up42-sentinel-poc` — same project
