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

# Super-Resolution
make wow             # WOW SR with auto-fetch
make sr              # Standard SR (x4)

# Utilities
make logs            # View container logs
make shell           # Open container shell
make clean           # Remove all data

# Status
make pipeline-status JOB=pipeline_20260102_123456
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
│   │   ├── main.py           # FastAPI application
│   │   ├── wow_sr.py         # WOW super-resolution
│   │   ├── smart_fetch.py    # Image fetching logic
│   │   ├── tiling.py         # Tile generation
│   │   └── cnn_super_resolution.py  # Real-ESRGAN
│   ├── Dockerfile
│   └── requirements.txt
├── data/
│   ├── source/               # Downloaded GeoTIFFs
│   ├── tiles/                # Original tiles
│   ├── tiles_sr/             # SR tiles
│   └── tiles_wow/            # WOW SR tiles
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
