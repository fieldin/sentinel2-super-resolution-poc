# Sentinel-2 Super-Resolution POC - Makefile
# ============================================
# One-click commands for building and running the POC

.PHONY: help up down fetch tile sr sr-tile clean poc poc-sr poc-clean build-client logs shell check-env

# Default target
help:
	@echo "Sentinel-2 Super-Resolution POC"
	@echo "================================"
	@echo ""
	@echo "Quick Start:"
	@echo "  make poc          - One-click: fetch → tile → up"
	@echo "  make poc-sr       - Full SR POC: fetch → sr → tile → up"
	@echo "  make poc-clean    - Clean restart"
	@echo ""
	@echo "Super-Resolution:"
	@echo "  make sr           - Apply AI Super-Resolution (x4 = 2.5m)"
	@echo "  make sr-x2        - Super-Resolution x2 (5m effective)"
	@echo "  make sr-tile      - Generate tiles from SR image"
	@echo ""
	@echo "Individual Commands:"
	@echo "  make up           - Start all services"
	@echo "  make down         - Stop all services"
	@echo "  make fetch        - Fetch Sentinel-2 imagery"
	@echo "  make tile         - Generate XYZ tiles"
	@echo "  make clean        - Remove all data files"
	@echo ""
	@echo "Development:"
	@echo "  make build-client - Build Angular client"
	@echo "  make logs         - View container logs"
	@echo "  make shell        - Open shell in server container"
	@echo ""
	@echo "Setup:"
	@echo "  1. Copy .env.example to .env"
	@echo "  2. Fill in UP42 and Mapbox credentials"
	@echo "  3. Run 'make poc' or 'make poc-sr'"
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
	rm -rf ./data/sr/*
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
