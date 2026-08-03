# sentinel2-super-resolution-poc — documentation index

POC for **AI super-resolution** of Sentinel-2 satellite imagery: fetch clearest recent scene, Real-ESRGAN x4 upscaling, XYZ tiles, optional field-boundary vectors, Angular + fld-map viewer.

## Doc index

| Doc | When to read |
|-----|----------------|
| [architecture-overview.md](./architecture-overview.md) | **Start here** — system diagram, data dirs, pipeline steps, resolution story |
| [pipeline-and-api-deep-dive.md](./pipeline-and-api-deep-dive.md) | FastAPI routes, job model, smart fetch, WOW SR, vectors, binary/job state |
| [client-and-deployment.md](./client-and-deployment.md) | Angular map + enhance UI, Docker, K8s, `deploy.sh`, staging URL |
| [cursor-ai-handoff.md](./cursor-ai-handoff.md) | Cursor agent rules, Makefile playbooks, debugging |

## Repo aliases

Folder: `sentinel2-super-resolution-poc`. Older notes / `gideline.txt` may say `up42-sentinel-poc` — same project.

## Staging

Public POC URL (after K8s deploy): `http://sentinel-poc-stg.fieldintech.com`

## Quick start

```bash
cp .env.example .env   # MAPBOX_ACCESS_TOKEN required; UP42 optional if using cached data
make up
make pipeline          # Fetch → tiles → WOW SR → SR tiles
open http://localhost:8080
```
