# Sentinel-2 POC - Combined Build
# Stage 1: Build Angular Client
FROM node:20-alpine AS client-builder

WORKDIR /client
COPY client/.npmrc ./
COPY client/package*.json ./
RUN npm ci
COPY client/ ./
RUN npm run build

# Stage 2: Python Server with Client Static Files
FROM python:3.11-slim-bookworm

# Install GDAL and dependencies for computer vision
RUN apt-get update && apt-get install -y --no-install-recommends \
  gdal-bin \
  libgdal-dev \
  python3-gdal \
  libgl1-mesa-glx \
  libglib2.0-0 \
  wget \
  git \
  curl \
  nginx \
  && rm -rf /var/lib/apt/lists/*

# Set GDAL environment
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Create app directory
WORKDIR /app

# Install PyTorch CPU version (required for Real-ESRGAN)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install Python dependencies
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server/app/ ./app/

# Create data and model directories
RUN mkdir -p /app/data/source /app/data/tiles /app/data/tiles_sr /app/data/tiles_wow \
  /app/data/sr /app/data/wow /app/data/vectors /app/config /app/static /app/models

# Copy built Angular client
COPY --from=client-builder /client/dist/sentinel-map/browser /app/static

# Copy config files
COPY config/ /app/config/

# Copy pre-generated tiles and vectors (excludes large .tif source files via .dockerignore)
COPY data/ /app/data/

# SSL certs and nginx config
COPY ./fieldin.crt /certs/fieldin.crt
COPY ./fieldin.key /certs/fieldin.key
COPY nginx.conf /etc/nginx/nginx.conf
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose HTTPS port
EXPOSE 443

# Health check against uvicorn directly
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

CMD ["/app/entrypoint.sh"]

