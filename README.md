# UP42 Sentinel-2 POC

A complete proof-of-concept for fetching, processing, and visualizing Sentinel-2 Surface Reflectance imagery using UP42, GDAL, and Mapbox GL JS.

![Architecture](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square)
![Architecture](https://img.shields.io/badge/Frontend-Angular%2018-DD0031?style=flat-square)
![Architecture](https://img.shields.io/badge/Mapping-Mapbox%20GL-4264FB?style=flat-square)
![Architecture](https://img.shields.io/badge/Tiling-GDAL-5CAE58?style=flat-square)

## 🎯 Features

- **Automated Imagery Fetch**: Search UP42 catalog for cloud-free Sentinel-2 L2A scenes
- **Smart Scene Selection**: Automatically selects best scene by cloud cover and date
- **XYZ Tile Generation**: GDAL-powered tile generation for web display
- **Interactive Map Viewer**: Angular + Mapbox GL JS with opacity controls
- **One-Click Deployment**: Docker Compose + Makefile orchestration

## 📁 Project Structure

```
up42-sentinel-poc/
├── server/
│   └── app/
│       ├── main.py           # FastAPI server
│       ├── settings.py       # Pydantic config
│       ├── up42_client.py    # UP42 API client
│       ├── fetch.py          # CLI: fetch imagery
│       ├── tile.py           # CLI: generate tiles
│       ├── tiling.py         # GDAL helpers
│       └── utils.py          # Logging, retries
├── client/                   # Angular application
│   └── src/app/
│       ├── map/              # Map component
│       └── services/         # Config & metadata services
├── config/
│   └── aoi.geojson          # Area of Interest
├── data/                     # Runtime outputs
│   ├── source/              # Downloaded GeoTIFFs
│   └── tiles/               # Generated XYZ tiles
├── docker-compose.yml
├── Makefile
└── .env.example
```

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- UP42 account with API credentials
- Mapbox account with access token

### Setup

1. **Clone and configure**:
   ```bash
   cd up42-sentinel-poc
   cp .env.example .env
   ```

2. **Edit `.env`** with your credentials:
   ```bash
   # UP42 Credentials (from console.up42.com > Project Settings > Developers)
   UP42_CLIENT_ID=your-client-id
   UP42_CLIENT_SECRET=your-client-secret
   UP42_PROJECT_ID=d73248f1-b99a-4466-b319-e0c923d5304d  # Your account ID
   
   # Mapbox Token (from account.mapbox.com)
   MAPBOX_ACCESS_TOKEN=pk.your-mapbox-token
   ```

3. **Configure AOI** (optional):
   Edit `config/aoi.geojson` with your target area polygon (EPSG:4326)

### One-Click POC

```bash
make poc
```

This will:
1. Fetch Sentinel-2 imagery from UP42
2. Generate XYZ tiles with GDAL
3. Start the server at http://localhost:8080

### Manual Steps

```bash
# Start containers
make up

# Fetch imagery (runs in container)
make fetch

# Generate tiles (runs in container)
make tile

# View logs
make logs
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UP42_CLIENT_ID` | required | UP42 OAuth2 Client ID |
| `UP42_CLIENT_SECRET` | required | UP42 OAuth2 Client Secret |
| `UP42_PROJECT_ID` | required | UP42 Project/Account ID |
| `DAYS_LOOKBACK` | `30` | Days to search back for imagery |
| `MAX_CLOUD_PCT` | `10` | Maximum cloud coverage % |
| `TILE_MIN_ZOOM` | `10` | Minimum tile zoom level |
| `TILE_MAX_ZOOM` | `16` | Maximum tile zoom level |
| `MAPBOX_ACCESS_TOKEN` | required | Mapbox GL JS access token |

### AOI Configuration

Edit `config/aoi.geojson` with a GeoJSON Polygon:

```json
{
  "type": "Feature",
  "properties": { "name": "My Farm" },
  "geometry": {
    "type": "Polygon",
    "coordinates": [[
      [-121.68, 36.62],
      [-121.60, 36.62],
      [-121.60, 36.68],
      [-121.68, 36.68],
      [-121.68, 36.62]
    ]]
  }
}
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/config` | GET | Client config (Mapbox token, zoom defaults) |
| `/api/metadata` | GET | Tileset + source metadata |
| `/tiles/{z}/{x}/{y}.png` | GET | XYZ raster tiles |

## 🗺️ Angular Client

The Angular client provides:

- **Map Display**: Mapbox GL JS with dark theme
- **Raster Overlay**: Sentinel-2 tiles with opacity control
- **Scene Info**: Acquisition date, cloud cover, file size
- **Zoom Controls**: Current zoom display, zoom-to-extent

### Development

```bash
cd client
npm install
npm start  # Runs on localhost:4200 with proxy to backend
```

## 🐛 Troubleshooting

### Authentication Errors

```
Error: 401 Unauthorized
```

- Verify `UP42_CLIENT_ID` and `UP42_CLIENT_SECRET` in `.env`
- Ensure credentials have catalog access permissions
- Check if credentials are for the correct environment (production vs sandbox)

### No Scenes Found

```
Error: No scenes found within 30 days with cloud cover <= 10%
```

- Increase `DAYS_LOOKBACK` (e.g., 60 or 90)
- Increase `MAX_CLOUD_PCT` (e.g., 20 or 30)
- Verify AOI coordinates are correct (should be in EPSG:4326)

### GDAL Errors

```
Error: gdal2tiles.py failed
```

- Check that the source GeoTIFF exists in `data/source/`
- Verify the file is not corrupted: `gdalinfo data/source/*.tif`
- Ensure sufficient disk space for tile generation

### Tiles Not Loading

- Check browser console for 404 errors
- Verify tiles exist: `ls data/tiles/`
- Check `data/tiles/tileset.json` has correct bounds
- Ensure Mapbox token is valid

### Mock Data Mode

If UP42 credentials are not configured, the system uses a mock client that generates placeholder data. You'll see:

```
UP42 credentials not configured - using mock client
```

This is useful for testing the tile generation and UI without UP42 access.

## 📦 Makefile Commands

| Command | Description |
|---------|-------------|
| `make poc` | One-click: fetch → tile → up |
| `make poc-clean` | Clean restart: down → clean → poc |
| `make up` | Start containers |
| `make down` | Stop containers |
| `make fetch` | Run fetch pipeline |
| `make tile` | Run tile generation |
| `make clean` | Remove data files |
| `make logs` | View container logs |
| `make shell` | Shell into server container |

## 🔒 Security Notes

- Never commit `.env` files with real credentials
- The `/api/config` endpoint exposes the Mapbox token (POC only)
- For production, implement proper token management

## 📝 TODOs

The UP42 client includes several `TODO` comments marking areas that may need adjustment based on actual API responses:

- Collection ID for Sentinel-2 L2A catalog search
- Order creation payload structure
- Asset download URL extraction
- Order status polling

These are implemented with best-guess defaults and should work, but may need tweaking based on your specific UP42 project configuration.

## 📄 License

MIT License - See LICENSE file for details.

