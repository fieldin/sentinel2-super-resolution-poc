# 🛰️ Sentinel-2 Super-Resolution POC

AI-powered super-resolution for Sentinel-2 satellite imagery with automated fetching, tile generation, and web visualization.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)
![Real-ESRGAN](https://img.shields.io/badge/AI-Real--ESRGAN-purple)

## ✨ Features

- **🔍 Smart Fetch**: Automatically finds the best Sentinel-2 image (lowest cloud cover, most recent)
- **🎨 AI Super-Resolution**: Real-ESRGAN x4 upscaling (10m → 2.5m effective resolution)
- **🌾 Crop Optimization**: Enhanced post-processing for agricultural imagery visibility
- **🗺️ XYZ Tiles**: Automatic tile generation for web mapping
- **🚀 One-Click Pipeline**: Fetch → Tiles → SR → SR Tiles in a single API call
- **🌐 Web Viewer**: Built-in Mapbox GL viewer for visualization
- **🗺️ Vector Intelligence**: Extract crisp field boundary polygons for high-zoom visualization (NEW!)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Pipeline API                           │
│                   POST /api/pipeline                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
┌───────────┐  ┌───────────┐  ┌───────────┐
│  Smart    │  │   Tile    │  │  WOW SR   │
│  Fetch    │  │ Generator │  │ (ESRGAN)  │
└─────┬─────┘  └─────┬─────┘  └─────┬─────┘
      │              │              │
      ▼              ▼              ▼
┌───────────┐  ┌───────────┐  ┌───────────┐
│ Sentinel-2│  │  /tiles/  │  │/tiles_sr/ │
│  GeoTIFF  │  │  z/x/y    │  │  z/x/y    │
└───────────┘  └───────────┘  └───────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Mapbox Access Token (for web viewer)

### Setup

1. **Clone and configure:**
   ```bash
   cd up42-sentinel-poc
   cp .env.example .env
   # Edit .env with your credentials
   ```

2. **Start the server:**
   ```bash
   make up
   ```

3. **Run the full pipeline:**
   ```bash
   make pipeline
   ```

4. **Open the viewer:**
   ```
   http://localhost:8080
   ```

## 📡 API Endpoints

### Full Pipeline (Recommended)

```bash
# Run complete flow: Fetch → Tiles → SR → SR Tiles
curl -X POST http://localhost:8080/api/pipeline \
  -H "Content-Type: application/json" \
  -d '{
    "max_age_days": 30,
    "max_cloud_cover": 30,
    "sr_type": "wow",
    "enhance_crops": true
  }'

# Check status
curl http://localhost:8080/api/pipeline/{job_id}

# List all jobs
curl http://localhost:8080/api/pipelines
```

### Individual Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/pipeline` | POST | Full pipeline (fetch + tiles + SR) |
| `/api/wow` | POST | WOW SR only (Real-ESRGAN x4 + Enhanced) |
| `/api/sr` | POST | Standard SR (EDSR/Farm SR) |
| `/api/metadata` | GET | Tileset and source metadata |
| `/tiles/{z}/{x}/{y}.png` | GET | Original tiles |
| `/tiles_sr/{z}/{x}/{y}.png` | GET | SR tiles |
| `/tiles_wow/{z}/{x}/{y}.png` | GET | WOW SR tiles |
| `/docs` | GET | Swagger API documentation |

## 🎯 Make Commands

```bash
# Quick Start
make up              # Start server
make pipeline        # Run full pipeline via API
make pipeline-fast   # Tiles only (no SR)
make pipeline-full   # Pipeline + vector extraction

# Super-Resolution
make wow             # WOW SR with auto-fetch
make sr              # Standard SR (x4)

# Vector Intelligence
make vectors         # Extract field boundary polygons
make vectors-api     # Extract via API (background)
make vectors-status  # Check vector extraction status

# Utilities
make logs            # View container logs
make shell           # Open container shell
make clean           # Remove all data

# Status
make pipeline-status JOB=pipeline_20260102_123456
make vectors-status JOB=vectors_20260102_123456
```

## 🔧 Configuration

### Environment Variables (.env)

```env
# Mapbox (required for web viewer)
MAPBOX_ACCESS_TOKEN=pk.xxx

# UP42 (optional, for fetching new imagery)
UP42_PROJECT_ID=xxx
UP42_PROJECT_API_KEY=xxx

# Tile settings
TILE_MIN_ZOOM=10
TILE_MAX_ZOOM=16
```

### Pipeline Options

```json
{
  "max_age_days": 30,           // How far back to search for images
  "max_cloud_cover": 30.0,      // Maximum cloud cover %
  "force_fetch": false,         // Force remote fetch
  "generate_original_tiles": true,
  "min_zoom": 10,
  "max_zoom": 16,
  "run_sr": true,               // Run super-resolution
  "sr_type": "wow",             // "wow" or "farm"
  "enhance_crops": true         // CLAHE + unsharp mask
}
```

## 🎨 Super-Resolution Pipeline

### WOW SR (Recommended for z18)

```
Input (10m) → Real-ESRGAN x4 → Post-Processing → Output (2.5m)
                    │
                    ├── CLAHE (local contrast)
                    ├── Unsharp mask (edge sharpening)
                    └── Vegetation boost (green enhancement)
```

**Optimized for:**
- Agricultural/crop imagery
- High zoom levels (z18)
- Crop row visibility

## 🗺️ Vector Intelligence

**NEW!** Extract crisp field boundary polygons for a Skycuse-like high-zoom experience.

### Why Vectors?

At zoom levels 18-20, raster imagery (even super-resolved) starts to show pixelation. Vector overlays provide:
- **Crisp boundaries** that scale perfectly at any zoom
- **Semantic information** (field IDs, areas, confidence scores)
- **Interactive features** (hover highlights, popups)
- **Better visual hierarchy** when raster fades slightly

### How It Works

```
Raster Input → Vegetation Mask → Segmentation → Polygons → GeoJSON
      │               │               │              │
      ├── SR output   ├── NDVI        ├── Watershed  ├── Simplify
      └── Original    └── HSV color   └── Morph ops  └── Clean topology
```

**Key features:**
- Processes entire AOI at once (no tile-breaking)
- Uses NDVI when spectral bands available
- Falls back to HSV color analysis for RGB images
- Topology cleanup (buffer(0), sliver removal, simplification)
- Confidence scoring based on shape and source

### Quick Start

```bash
# After running the pipeline:
make vectors

# Or via API:
curl -X POST http://localhost:8080/api/vectors -H "Content-Type: application/json"

# Check status:
make vectors-status
```

### Vector API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/vectors/fields.geojson` | GET | Serve field polygons as GeoJSON |
| `/api/vectors` | POST | Start vector extraction |
| `/api/vectors/{job_id}` | GET | Check extraction status |
| `/api/vectors/metadata` | GET | Get vector layer metadata |

### Frontend Features

- **Toggle control**: Show/hide field boundaries
- **Zoom-dependent styling**:
  - Lines thicken at higher zoom (1px → 4px)
  - Subtle fill appears at z17+
  - Raster fades slightly at z18+ to let vectors dominate
- **Hover interaction**: Highlight field + popup with area, confidence
- **Smooth animations**: Transition effects on hover

### Configuration Options

```bash
# Custom extraction parameters
python -m app.generate_vectors \
  --aoi config/aoi.geojson \
  --ndvi-threshold 0.25 \
  --min-area 0.5 \
  --max-area 200 \
  --simplify 3.0 \
  --out data/vectors
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ndvi-threshold` | 0.3 | NDVI threshold for vegetation |
| `--min-area` | 0.1 | Minimum field area (hectares) |
| `--max-area` | 500 | Maximum field area (hectares) |
| `--simplify` | 5.0 | Simplification tolerance (meters) |

### Output Format

The output GeoJSON (`data/vectors/fields.geojson`) contains:

```json
{
  "type": "FeatureCollection",
  "properties": {
    "generated_at": "2026-01-04T...",
    "source_method": "ndvi",
    "feature_count": 42
  },
  "features": [
    {
      "type": "Feature",
      "id": "abc123def456",
      "properties": {
        "id": "abc123def456",
        "field_index": 1,
        "area_ha": 12.5,
        "source": "ndvi",
        "confidence": 0.85,
        "created_at": "2026-01-04T..."
      },
      "geometry": { "type": "Polygon", "coordinates": [...] }
    }
  ]
}
```

## 📊 Resolution Comparison

| Source | Native Resolution | After SR | Zoom Level |
|--------|------------------|----------|------------|
| Sentinel-2 | 10m | 2.5m (x4) | z15-18 |
| Output Tiles | - | 256px | z10-18 |

## 📁 Project Structure

```
up42-sentinel-poc/
├── server/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── wow_sr.py            # WOW super-resolution
│   │   ├── smart_fetch.py       # Image fetching logic
│   │   ├── tiling.py            # Tile generation
│   │   ├── cnn_super_resolution.py  # Real-ESRGAN
│   │   ├── vector_extraction.py # Field polygon extraction (NEW!)
│   │   └── generate_vectors.py  # Vector CLI command (NEW!)
│   ├── Dockerfile
│   └── requirements.txt
├── client/
│   └── src/
│       └── app/
│           └── map/             # Mapbox GL map component
├── config/
│   └── aoi.geojson             # Area of interest definition
├── data/
│   ├── source/                 # Downloaded GeoTIFFs
│   ├── tiles/                  # Original tiles
│   ├── tiles_sr/               # SR tiles
│   ├── tiles_wow/              # WOW SR tiles
│   └── vectors/                # Field boundary vectors (NEW!)
│       └── fields.geojson
├── docker-compose.yml
├── Makefile
└── README.md
```

## 🔗 Data Sources

- **Sentinel-2**: AWS Earth Search STAC (free, public)
- **Fallback**: UP42 API (requires credentials)

## 📝 License

MIT License - See LICENSE file for details.

## 🙏 Credits

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - AI Super-Resolution
- [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) - Satellite Imagery
- [Mapbox GL JS](https://docs.mapbox.com/mapbox-gl-js/) - Web Mapping
