# Sentinel-2 Super-Resolution POC - Makefile
# ============================================
# One-click commands for building and running the POC

.PHONY: help up down fetch tile sr sr-tile clean poc poc-sr poc-clean build-client logs shell check-env pipeline pipeline-status vectors vectors-api vectors-status

# Default target
help:
	@echo "Sentinel-2 Super-Resolution POC"
	@echo "================================"
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make pipeline     - ONE API: Fetch → Tiles → SR → SR Tiles (NEW!)"
	@echo "  make poc          - One-click: fetch → tile → up"
	@echo "  make poc-sr       - Full SR POC: fetch → sr → tile → up"
	@echo "  make poc-clean    - Clean restart"
	@echo ""
	@echo "📡 Full Pipeline API:"
	@echo "  make pipeline           - Run complete pipeline via API"
	@echo "  make pipeline-fast      - Pipeline without SR (quick tiles only)"
	@echo "  make pipeline-status    - Check pipeline job status"
	@echo ""
	@echo "🎨 Super-Resolution:"
	@echo "  make sr           - Apply AI Super-Resolution (x4 = 2.5m)"
	@echo "  make sr-x2        - Super-Resolution x2 (5m effective)"
	@echo "  make sr-tile      - Generate tiles from SR image"
	@echo "  make wow          - WOW SR with auto-fetch (best for z18!) 🌟"
	@echo "  make wow-file     - WOW SR on specific file"
	@echo "  make smart-fetch  - Auto-get best image (last 30d, cloud ≤30%)"
	@echo ""
	@echo "🗺️ Vector Intelligence:"
	@echo "  make vectors      - Extract field boundary polygons 🗺️"
	@echo "  make vectors-v2   - V2: gradient segmentation + OSM roads + zones 🚀"
	@echo "  make multiband    - Fetch & stack Sentinel-2 B04+B08+SCL (real NDVI) 📥"
	@echo "  make vectors-ndvi - multiband → vectors-v2 (source=ndvi_bands) ✅"
	@echo "  make vectors-api  - Extract vectors via API"
	@echo "  make vectors-status - Check vector extraction status"
	@echo ""
	@echo "⚙️ Individual Commands:"
	@echo "  make up           - Start all services"
	@echo "  make down         - Stop all services"
	@echo "  make fetch        - Fetch Sentinel-2 imagery"
	@echo "  make tile         - Generate XYZ tiles"
	@echo "  make clean        - Remove all data files"
	@echo ""
	@echo "🛠️ Development:"
	@echo "  make build-client - Build Angular client"
	@echo "  make logs         - View container logs"
	@echo "  make shell        - Open shell in server container"
	@echo ""
	@echo "Setup:"
	@echo "  1. Copy .env.example to .env"
	@echo "  2. Fill in UP42 and Mapbox credentials"
	@echo "  3. Run 'make pipeline' (recommended) or 'make poc'"
	@echo ""

# Build and start containers
up:
	@echo "Building and starting containers..."
	docker compose up --build -d
	@echo ""
	@echo "Server running at http://localhost:8080"
	@echo ""

# Stop containers
down:
	@echo "Stopping containers..."
	docker compose down

# Fetch Sentinel-2 imagery via UP42
fetch:
	@echo "Fetching Sentinel-2 imagery..."
	@if docker compose ps -q server > /dev/null 2>&1 && docker compose ps | grep -q "Up"; then \
		docker compose exec server python -m app.fetch; \
	else \
		docker compose run --rm server python -m app.fetch; \
	fi

# Generate XYZ tiles from downloaded imagery
tile:
	@echo "Generating XYZ tiles..."
	@if docker compose ps -q server > /dev/null 2>&1 && docker compose ps | grep -q "Up"; then \
		docker compose exec server python -m app.tile; \
	else \
		docker compose run --rm server python -m app.tile; \
	fi

# Apply AI Super-Resolution (x4 = 2.5m effective resolution)
sr:
	@echo "Applying AI Super-Resolution (x4)..."
	@echo "This may take a few minutes..."
	@if docker compose ps -q server > /dev/null 2>&1 && docker compose ps | grep -q "Up"; then \
		docker compose exec server python -m app.sr_cli --scale 4; \
	else \
		docker compose run --rm server python -m app.sr_cli --scale 4; \
	fi

# Apply AI Super-Resolution (x2 = 5m effective resolution)
sr-x2:
	@echo "Applying AI Super-Resolution (x2)..."
	@if docker compose ps -q server > /dev/null 2>&1 && docker compose ps | grep -q "Up"; then \
		docker compose exec server python -m app.sr_cli --scale 2; \
	else \
		docker compose run --rm server python -m app.sr_cli --scale 2; \
	fi

# Smart Fetch: Get best image (latest + clearest from last 30 days)
smart-fetch:
	@echo "Smart Fetch: Finding best Sentinel-2 image..."
	@echo "(Last 30 days, cloud ≤30%, fetches from UP42 only if needed)"
	@if docker compose ps -q server > /dev/null 2>&1 && docker compose ps | grep -q "Up"; then \
		docker compose exec server python -m app.smart_fetch; \
	else \
		docker compose run --rm server python -m app.smart_fetch; \
	fi

# WOW Super-Resolution: SwinIR x2 → Real-ESRGAN x2 (best for z18)
# Automatically fetches best image if needed!
wow:
	@echo "WOW Super-Resolution (SwinIR x2 → Real-ESRGAN x2)..."
	@echo "Auto-fetches best image (last 30 days, cloud ≤30%)"
	@echo "This provides superior quality at z18!"
	@curl -s -X POST http://localhost:8080/api/wow \
		-H "Content-Type: application/json" \
		-d '{"auto_fetch": true, "max_age_days": 30, "max_cloud_cover": 30}' | jq .

# WOW with specific file
wow-file:
	@echo "Usage: make wow-file FILE=data/source/your_image.tif"
	@if [ -z "$(FILE)" ]; then echo "Error: FILE not specified"; exit 1; fi
	@if docker compose ps -q server > /dev/null 2>&1 && docker compose ps | grep -q "Up"; then \
		docker compose exec server python -m app.wow_sr $(FILE) -o data/wow; \
	else \
		docker compose run --rm server python -m app.wow_sr $(FILE) -o data/wow; \
	fi

# Generate tiles from SR image
sr-tile:
	@echo "Generating tiles from SR image..."
	@if docker compose ps -q server > /dev/null 2>&1 && docker compose ps | grep -q "Up"; then \
		docker compose exec server python -m app.sr_cli --scale 4 --tile --tile-max-zoom 20; \
	else \
		docker compose run --rm server python -m app.sr_cli --scale 4 --tile --tile-max-zoom 20; \
	fi

# Clean all data files
clean:
	@echo "Cleaning data directories..."
	rm -rf ./data/source/*
	rm -rf ./data/tiles/*
	rm -rf ./data/tiles_sr/*
	rm -rf ./data/tiles_wow/*
	rm -rf ./data/sr/*
	rm -rf ./data/wow/*
	@echo "Data cleaned."

# One-click POC: fetch imagery, generate tiles, start server
poc: check-env create-minimal-client
	@echo "============================================"
	@echo "Sentinel-2 POC - One-Click Deployment"
	@echo "============================================"
	@echo ""
	@echo "Step 1/3: Fetching Sentinel-2 imagery..."
	@$(MAKE) fetch
	@echo ""
	@echo "Step 2/3: Generating XYZ tiles..."
	@$(MAKE) tile
	@echo ""
	@echo "Step 3/3: Starting server..."
	@$(MAKE) up
	@echo ""
	@echo "============================================"
	@echo "POC Ready!"
	@echo "Open http://localhost:8080 in your browser"
	@echo "============================================"

# Full Super-Resolution POC: fetch → SR → tile → up
poc-sr: check-env create-minimal-client
	@echo "============================================"
	@echo "Sentinel-2 SUPER-RESOLUTION POC"
	@echo "============================================"
	@echo ""
	@echo "Step 1/4: Fetching Sentinel-2 imagery..."
	@$(MAKE) fetch
	@echo ""
	@echo "Step 2/4: Applying AI Super-Resolution (x4)..."
	@$(MAKE) sr
	@echo ""
	@echo "Step 3/4: Generating XYZ tiles from SR..."
	@$(MAKE) sr-tile
	@echo ""
	@echo "Step 4/4: Starting server..."
	@$(MAKE) up
	@echo ""
	@echo "============================================"
	@echo "Super-Resolution POC Ready!"
	@echo "Effective resolution: 2.5m (from 10m)"
	@echo "Open http://localhost:8080 in your browser"
	@echo "============================================"

# Clean and rebuild everything
poc-clean:
	@echo "Performing clean POC restart..."
	@$(MAKE) down
	@$(MAKE) clean
	@$(MAKE) poc

# Build Angular client
build-client:
	@echo "Building Angular client..."
	@if [ -d "./client/node_modules" ]; then \
		cd client && npm run build; \
	else \
		cd client && npm ci && npm run build; \
	fi

# Create minimal HTML client if Angular build not available
create-minimal-client:
	@mkdir -p ./client/dist/sentinel-map
	@if [ ! -f "./client/dist/sentinel-map/index.html" ]; then \
		echo "Creating minimal HTML viewer..."; \
		$(MAKE) write-minimal-html; \
	fi

# Write minimal HTML (separate target for cleaner code)
write-minimal-html:
	@echo '<!DOCTYPE html>' > ./client/dist/sentinel-map/index.html
	@echo '<html lang="en">' >> ./client/dist/sentinel-map/index.html
	@echo '<head>' >> ./client/dist/sentinel-map/index.html
	@echo '  <meta charset="UTF-8">' >> ./client/dist/sentinel-map/index.html
	@echo '  <meta name="viewport" content="width=device-width, initial-scale=1.0">' >> ./client/dist/sentinel-map/index.html
	@echo '  <title>Sentinel-2 Viewer</title>' >> ./client/dist/sentinel-map/index.html
	@echo '  <script src="https://api.mapbox.com/mapbox-gl-js/v3.0.1/mapbox-gl.js"></script>' >> ./client/dist/sentinel-map/index.html
	@echo '  <link href="https://api.mapbox.com/mapbox-gl-js/v3.0.1/mapbox-gl.css" rel="stylesheet">' >> ./client/dist/sentinel-map/index.html
	@echo '  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400&family=Outfit:wght@400;500&display=swap" rel="stylesheet">' >> ./client/dist/sentinel-map/index.html
	@echo '  <style>' >> ./client/dist/sentinel-map/index.html
	@echo '    :root{--bg:#0a0e17;--card:rgba(26,35,50,0.95);--accent:#00d4aa;--text:#f1f5f9;--muted:#64748b;--border:#2d3a4f}' >> ./client/dist/sentinel-map/index.html
	@echo '    *{margin:0;padding:0;box-sizing:border-box}body{font-family:Outfit,sans-serif;background:var(--bg);color:var(--text)}' >> ./client/dist/sentinel-map/index.html
	@echo '    #map{position:absolute;inset:0}' >> ./client/dist/sentinel-map/index.html
	@echo '    .panel{position:absolute;top:16px;left:16px;width:280px;background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px;backdrop-filter:blur(12px);z-index:10}' >> ./client/dist/sentinel-map/index.html
	@echo '    .panel h1{font-size:16px;font-weight:500;margin-bottom:16px;display:flex;align-items:center;gap:8px}' >> ./client/dist/sentinel-map/index.html
	@echo '    .panel h1::before{content:"🛰️"}' >> ./client/dist/sentinel-map/index.html
	@echo '    .status{font-size:13px;padding:8px 0;border-bottom:1px solid var(--border);margin-bottom:12px}' >> ./client/dist/sentinel-map/index.html
	@echo '    .status.ready{color:var(--accent)}.status.error{color:#ef4444}' >> ./client/dist/sentinel-map/index.html
	@echo '    label{display:flex;justify-content:space-between;font-size:12px;color:var(--muted);margin-bottom:4px}' >> ./client/dist/sentinel-map/index.html
	@echo '    label span:last-child{color:var(--text)}' >> ./client/dist/sentinel-map/index.html
	@echo '    input[type=range]{width:100%;margin:8px 0 16px;accent-color:var(--accent)}' >> ./client/dist/sentinel-map/index.html
	@echo '    .info{font-size:11px;color:var(--muted);border-top:1px solid var(--border);padding-top:12px;margin-top:8px}' >> ./client/dist/sentinel-map/index.html
	@echo '    .info div{display:flex;justify-content:space-between;padding:4px 0}' >> ./client/dist/sentinel-map/index.html
	@echo '    .info span:last-child{color:var(--text);font-family:"JetBrains Mono",monospace;font-size:10px}' >> ./client/dist/sentinel-map/index.html
	@echo '  </style>' >> ./client/dist/sentinel-map/index.html
	@echo '</head>' >> ./client/dist/sentinel-map/index.html
	@echo '<body>' >> ./client/dist/sentinel-map/index.html
	@echo '  <div id="map"></div>' >> ./client/dist/sentinel-map/index.html
	@echo '  <div class="panel">' >> ./client/dist/sentinel-map/index.html
	@echo '    <h1>Sentinel-2 Viewer</h1>' >> ./client/dist/sentinel-map/index.html
	@echo '    <div class="status" id="status">Loading...</div>' >> ./client/dist/sentinel-map/index.html
	@echo '    <label><span>Opacity</span><span id="opval">80%</span></label>' >> ./client/dist/sentinel-map/index.html
	@echo '    <input type="range" id="opacity" min="0" max="100" value="80">' >> ./client/dist/sentinel-map/index.html
	@echo '    <label><span>Zoom</span><span id="zoom">-</span></label>' >> ./client/dist/sentinel-map/index.html
	@echo '    <div class="info" id="info"></div>' >> ./client/dist/sentinel-map/index.html
	@echo '  </div>' >> ./client/dist/sentinel-map/index.html
	@echo '  <script>' >> ./client/dist/sentinel-map/index.html
	@echo '    (async()=>{const s=document.getElementById("status"),i=document.getElementById("info");try{const[c,m]=await Promise.all([fetch("/api/config").then(r=>r.json()),fetch("/api/metadata").then(r=>r.json())]);mapboxgl.accessToken=c.mapboxAccessToken;const map=new mapboxgl.Map({container:"map",style:"mapbox://styles/mapbox/dark-v11",center:c.defaultCenter,zoom:c.defaultZoom});map.addControl(new mapboxgl.NavigationControl(),"top-right");map.on("load",()=>{if(m.tilesAvailable&&m.tileset){const t=m.tileset;map.addSource("sentinel",{type:"raster",tiles:[location.origin+t.tileTemplate],tileSize:256,bounds:t.bounds,minzoom:t.minzoom,maxzoom:t.maxzoom});map.addLayer({id:"sentinel-layer",type:"raster",source:"sentinel",paint:{"raster-opacity":0.8}});map.fitBounds([[t.bounds[0],t.bounds[1]],[t.bounds[2],t.bounds[3]]],{padding:50});s.textContent="Tiles loaded";s.className="status ready"}else{s.textContent="No tiles - run fetch & tile";s.className="status error"}if(m.source){i.innerHTML="<div><span>Scene</span><span>"+(m.source.scene_id||"").slice(0,15)+"...</span></div><div><span>Date</span><span>"+new Date(m.source.acquisition_date).toLocaleDateString()+"</span></div><div><span>Cloud</span><span>"+m.source.cloud_cover_pct+"%</span></div>"}});map.on("zoom",()=>document.getElementById("zoom").textContent=map.getZoom().toFixed(1));document.getElementById("opacity").oninput=e=>{document.getElementById("opval").textContent=e.target.value+"%";map.getLayer("sentinel-layer")&&map.setPaintProperty("sentinel-layer","raster-opacity",e.target.value/100)}}catch(e){s.textContent="Error: "+e.message;s.className="status error"}})();' >> ./client/dist/sentinel-map/index.html
	@echo '  </script>' >> ./client/dist/sentinel-map/index.html
	@echo '</body></html>' >> ./client/dist/sentinel-map/index.html

# Check environment variables
check-env:
	@if [ ! -f .env ]; then \
		echo "Error: .env file not found!"; \
		echo "Copy .env.example to .env and fill in your credentials."; \
		exit 1; \
	fi

# View logs
logs:
	docker compose logs -f

# Open shell in server container
shell:
	docker compose exec server /bin/bash

# ============================================================
# FULL PIPELINE API
# ============================================================

# Run complete pipeline: Fetch → Tiles → WOW SR → SR Tiles
pipeline:
	@echo "============================================"
	@echo "🚀 Running Full Pipeline via API"
	@echo "============================================"
	@echo ""
	@echo "Pipeline: Fetch → Tiles → WOW SR → SR Tiles"
	@echo ""
	@curl -s -X POST http://localhost:8080/api/pipeline \
		-H "Content-Type: application/json" \
		-d '{"max_age_days": 30, "max_cloud_cover": 30, "sr_type": "wow", "enhance_crops": true}' | jq .
	@echo ""
	@echo "Monitor progress: make pipeline-status JOB=<job_id>"

# Pipeline without SR (quick tiles only)
pipeline-fast:
	@echo "🚀 Running Fast Pipeline (no SR)..."
	@curl -s -X POST http://localhost:8080/api/pipeline \
		-H "Content-Type: application/json" \
		-d '{"run_sr": false}' | jq .

# Pipeline with Farm SR instead of WOW
pipeline-farm:
	@echo "🚀 Running Pipeline with Farm SR..."
	@curl -s -X POST http://localhost:8080/api/pipeline \
		-H "Content-Type: application/json" \
		-d '{"sr_type": "farm"}' | jq .

# Check pipeline status
pipeline-status:
	@if [ -z "$(JOB)" ]; then \
		echo "Listing all pipeline jobs..."; \
		curl -s http://localhost:8080/api/pipelines | jq .; \
	else \
		echo "Status for job: $(JOB)"; \
		curl -s http://localhost:8080/api/pipeline/$(JOB) | jq .; \
	fi

# Watch pipeline progress (polls every 5 seconds)
pipeline-watch:
	@if [ -z "$(JOB)" ]; then \
		echo "Error: JOB not specified. Usage: make pipeline-watch JOB=pipeline_20260102_123456"; \
		exit 1; \
	fi
	@echo "Watching pipeline job: $(JOB)"
	@echo "Press Ctrl+C to stop"
	@echo ""
	@while true; do \
		clear; \
		echo "Pipeline: $(JOB)"; \
		echo "==========================================="; \
		curl -s http://localhost:8080/api/pipeline/$(JOB) | jq '{status, current_step, message, steps: [.steps[]? | {step, name, status, message}]}'; \
		STATUS=$$(curl -s http://localhost:8080/api/pipeline/$(JOB) | jq -r '.status'); \
		if [ "$$STATUS" = "completed" ] || [ "$$STATUS" = "failed" ]; then \
			echo ""; \
			echo "Pipeline $$STATUS!"; \
			break; \
		fi; \
		sleep 5; \
	done

# ============================================================
# VECTOR INTELLIGENCE
# ============================================================

# Extract field boundary polygons from satellite imagery
# This creates data/vectors/fields.geojson which is served by the API
vectors:
	@echo "============================================"
	@echo "🗺️  VECTOR INTELLIGENCE - Field Extraction"
	@echo "============================================"
	@echo ""
	@echo "Extracting field boundaries from satellite imagery..."
	@echo "This uses NDVI or HSV color analysis to detect fields."
	@echo ""
	@if docker compose ps -q server > /dev/null 2>&1 && docker compose ps | grep -q "Up"; then \
		docker compose exec server python -m app.generate_vectors; \
	else \
		docker compose run --rm server python -m app.generate_vectors; \
	fi

# V2 Enhanced vector extraction with:
# - Gradient-based segmentation (better boundary detection)
# - OSM road clipping (splits fields at roads)
# - Management zones (k-means clustering)
# - Rich per-field statistics (mean_ndvi, ndvi_std, edge_strength)
vectors-v2:
	@echo "============================================"
	@echo "🚀 VECTOR INTELLIGENCE V2 - Enhanced"
	@echo "============================================"
	@echo ""
	@echo "Features:"
	@echo "  • Gradient-based segmentation (Sobel → watershed)"
	@echo "  • OSM road clipping (field splitting)"
	@echo "  • Management zones (k-means on NDVI)"
	@echo "  • Rich statistics (mean_ndvi, ndvi_std, edge_strength)"
	@echo ""
	@if docker compose ps -q server > /dev/null 2>&1 && docker compose ps | grep -q "Up"; then \
		docker compose exec server python -m app.vector_extraction_v2 \
			--aoi config/aoi.geojson \
			--rasters data/source/*.tif \
			--out data/vectors; \
	else \
		docker compose run --rm server python -m app.vector_extraction_v2 \
			--aoi config/aoi.geojson \
			--rasters data/source/*.tif \
			--out data/vectors; \
	fi

# Fetch & stack Sentinel-2 bands for REAL NDVI (B04+B08) + SCL
multiband:
	@echo "============================================"
	@echo "📥 Fetching Sentinel-2 multiband (B04+B08+SCL)"
	@echo "============================================"
	@if docker compose ps -q server > /dev/null 2>&1 && docker compose ps | grep -q "Up"; then \
		docker compose exec server python -m app.fetch_multiband --aoi config/aoi.geojson --out data/source; \
	else \
		docker compose run --rm server python -m app.fetch_multiband --aoi config/aoi.geojson --out data/source; \
	fi

# One-shot: fetch multiband then run vectors-v2 using that stack
vectors-ndvi: multiband
	@echo "============================================"
	@echo "✅ Extracting vectors from real NDVI (B04+B08)"
	@echo "============================================"
	@if docker compose ps -q server > /dev/null 2>&1 && docker compose ps | grep -q "Up"; then \
		docker compose exec server sh -lc 'python -m app.vector_extraction_v2 \
			--aoi config/aoi.geojson \
			--rasters data/source/s2_multiband_*.tif \
			--out data/vectors \
			--no-osm --no-zones'; \
	else \
		docker compose run --rm server sh -lc 'python -m app.vector_extraction_v2 \
			--aoi config/aoi.geojson \
			--rasters data/source/s2_multiband_*.tif \
			--out data/vectors \
			--no-osm --no-zones'; \
	fi

# Extract vectors with custom AOI
vectors-custom:
	@echo "Usage: make vectors-custom AOI=path/to/aoi.geojson"
	@if [ -z "$(AOI)" ]; then echo "Error: AOI not specified"; exit 1; fi
	@if docker compose ps -q server > /dev/null 2>&1 && docker compose ps | grep -q "Up"; then \
		docker compose exec server python -m app.generate_vectors --aoi $(AOI); \
	else \
		docker compose run --rm server python -m app.generate_vectors --aoi $(AOI); \
	fi

# Extract vectors via API (runs in background)
vectors-api:
	@echo "🗺️  Starting vector extraction via API..."
	@curl -s -X POST http://localhost:8080/api/vectors \
		-H "Content-Type: application/json" \
		-d '{"ndvi_threshold": 0.3, "min_area_ha": 0.1, "max_area_ha": 500}' | jq .
	@echo ""
	@echo "Monitor progress: make vectors-status JOB=<job_id>"

# Check vector extraction status
vectors-status:
	@if [ -z "$(JOB)" ]; then \
		echo "Checking vector metadata..."; \
		curl -s http://localhost:8080/api/vectors/metadata | jq .; \
	else \
		echo "Status for job: $(JOB)"; \
		curl -s http://localhost:8080/api/vectors/$(JOB) | jq .; \
	fi

# Full pipeline with vectors: Fetch → Tiles → SR → SR Tiles → Vectors
pipeline-full:
	@echo "============================================"
	@echo "🚀 Running Full Pipeline + Vectors"
	@echo "============================================"
	@echo ""
	@echo "Step 1: Running pipeline (Fetch → Tiles → SR → SR Tiles)..."
	@$(MAKE) pipeline
	@echo ""
	@echo "Waiting for pipeline to complete..."
	@sleep 10
	@echo ""
	@echo "Step 2: Extracting field boundary vectors..."
	@$(MAKE) vectors
	@echo ""
	@echo "============================================"
	@echo "🎉 Full Pipeline + Vectors Complete!"
	@echo "============================================"
	@echo ""
	@echo "View at: http://localhost:8080"
	@echo "Toggle 'Field Boundaries' to see vector overlay"
	@echo ""
