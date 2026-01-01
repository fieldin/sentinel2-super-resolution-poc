# Sentinel-2 Super-Resolution POC - Server Container
# Python 3.11 with GDAL, PyTorch, and OpenCV for AI Super-Resolution

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
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Create app directory
WORKDIR /app

# Install PyTorch CPU version (required for Real-ESRGAN)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create data and model directories
RUN mkdir -p /app/data/source /app/data/tiles /app/data/tiles_sr /app/data/sr \
    /app/config /app/static /app/models

# Expose port
EXPOSE 8080

# Default command - run server
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

