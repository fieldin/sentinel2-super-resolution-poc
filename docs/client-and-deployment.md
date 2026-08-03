# Client and deployment

## Angular client (`client/`)

Angular 20 standalone app. Built output: `client/dist/sentinel-map/` → copied to `/app/static` in Docker.

### Routes (`app.routes.ts`)

| Path | Component | Purpose |
|------|-----------|---------|
| `/` | `MapComponent` | Main SR map viewer |
| `/enhance` | `EnhanceComponent` | Upload + Real-ESRGAN enhance UI |

### Map viewer (`map.component.ts`)

Uses **fieldin-web-20** `FldMapModule` (fld-map) — same stack as fld-app-new storybook patterns.

Features:

- Loads `ConfigService.getConfig()` + `MetadataService.getMetadata()` on init.
- Adds **Leaflet tile layer** on top of fld-map base (Google/Mapbox satellite).
- Toggle **original** vs **WOW** tiles (`useWowTiles` signal).
- Opacity slider, zoom readout, acquisition metadata panel.
- **Polygon draw/edit** via fld-map `editPolygons` + `mapControlsService.addEditPolygonsControls()`.
- Custom cancel handler preserves drawn polygons in `drawnPolygonsLayer`.
- Export drawn polygons as GeoJSON download.
- User location marker (hardcoded JV Farms area coords in `USER_LOCATION`).

Tile URL resolution:

```typescript
// wow: /tiles_wow/{z}/{x}/{y}.png
// original: /tiles/{z}/{x}/{y}.png
// from metadata.tileEndpoints
```

`maxNativeZoom` from tileset; `maxZoom: 22` for overzoom.

### Enhance page (`enhance.component.ts`)

- Upload image → `EsrganService` → `POST /api/enhance`.
- Model picker: `realesrgan_x4`, `realesrgan_anime`.
- Progress stages: upload → processing → download.
- Optional QR/OCR via `QrOcrService` (client-side experiments).

### Services

| Service | File | API |
|---------|------|-----|
| `ConfigService` | `config.service.ts` | `GET /api/config` |
| `MetadataService` | `metadata.service.ts` | `GET /api/metadata` |
| `EsrganService` | `esrgan.service.ts` | `POST /api/enhance`, poll `/api/sr/{id}` |

### Local client dev

```bash
cd client
npm install
npm start   # proxy.conf.json → localhost:8080 API
```

Or use full stack: `make up` serves built static from container.

### Map keys note

`map.component.ts` embeds Mapbox/Google/Apple keys in `mapConfiguration.mapKeys` for fld-map. Server also exposes Mapbox via `/api/config`. **POC only** — do not copy keys to production apps without review.

## Docker

### docker-compose.yml

Services: `server` (app), optional local dev. Port **8080**.

Data bind mount: `./data` → `/app/data`.

### Root Dockerfile

Multi-stage:

1. Build Angular client (or use prebuilt dist).
2. Python 3.11 + `server/requirements.txt` + Real-ESRGAN deps.
3. `entrypoint.sh` — uvicorn on 8080 (nginx optional via `SKIP_NGINX`).

### Local docker (from `gideline.txt`)

```bash
docker rm -f up42-poc 2>/dev/null || true
docker build -t up42-poc:local -f Dockerfile .
docker run -d -p 8080:8080 --name up42-poc --env-file .env \
  -v $(pwd)/data:/app/data up42-poc:local
```

## Kubernetes staging

### deploy.sh

```bash
./deploy.sh latest           # full: client build + docker + ECR push + k8s apply
./deploy.sh latest --skip-build   # skip npm build
```

Steps:

1. `npm run build` in `client/`
2. `docker buildx build` → `838148646721.dkr.ecr.eu-west-1.amazonaws.com/sentinel-poc:latest`
3. Push to ECR (`AWS_PROFILE=awsnew`)
4. `kubectl apply -f k8s/` (namespace `sentinel-poc`)
5. Rollout restart + health check

### K8s resources (`k8s/`)

| File | Resource |
|------|----------|
| `namespace.yaml` | `sentinel-poc` namespace |
| `deployment.yaml` | 1 replica, 512Mi–2Gi RAM, port 8080 |
| `service.yaml` | ClusterIP |
| `service-public.yaml` | Public exposure |
| `ingress.yaml` | ALB → `sentinel-poc-stg.fieldintech.com` |

Secrets: `sentinel-poc-secrets` (UP42 optional). Mapbox token currently in deployment env (POC).

### Staging URL

```
http://sentinel-poc-stg.fieldintech.com
GET /health
```

**Data persistence:** Confirm whether staging pod mounts EFS/volume for `data/` — without it, pipeline must be re-run after each deploy.

## Makefile cheat sheet

| Command | Action |
|---------|--------|
| `make up` | docker compose up |
| `make pipeline` | POST full pipeline |
| `make pipeline-watch JOB=...` | Poll job status |
| `make pipeline-fast` | Tiles only, no SR |
| `make pipeline-full` | Pipeline + vectors |
| `make wow` | WOW SR via API |
| `make vectors` / `vectors-v2` | Field polygons |
| `make vectors-ndvi` | Multiband fetch + NDVI vectors |
| `make multiband` | B04+B08+SCL stack |
| `make clean` | Wipe `data/` subdirs |
| `make logs` / `make shell` | Debug container |

## nginx

`nginx.conf` / `client/nginx-proxy.conf` — used when nginx fronting enabled. K8s sets `SKIP_NGINX=true` — uvicorn serves directly.

## Pre-loaded sample data

Repo may include sample outputs under `data/` (tiles_wow, sr metadata). Allows UI demo without live fetch when UP42/AWS unavailable.

Check `data/tiles_wow/tileset.json` and `data/source/*_meta.json` for scene date and bounds.
