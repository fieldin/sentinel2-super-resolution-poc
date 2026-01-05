#!/bin/bash
set -e

REGISTRY="registry.fieldintech.com:5000"
IMAGE_NAME="fieldin/sentinel-poc"
TAG="${1:-latest}"

echo "🚀 Deploying Sentinel-2 POC to Kubernetes"
echo "=========================================="

# Step 1: Build the combined Docker image
echo ""
echo "📦 Step 1: Building Docker image..."
cd /home/michael/fieldin/up42-sentinel-poc

# Build combined image (client + server + data)
docker build -t ${REGISTRY}/${IMAGE_NAME}:${TAG} .

echo "✅ Image built: ${REGISTRY}/${IMAGE_NAME}:${TAG}"

# Step 2: Push to registry
echo ""
echo "📤 Step 2: Pushing to registry..."
docker push ${REGISTRY}/${IMAGE_NAME}:${TAG}
echo "✅ Image pushed"

# Step 3: Apply Kubernetes manifests
echo ""
echo "🎯 Step 3: Applying Kubernetes manifests..."

# Create namespace if not exists
kubectl apply -f k8s/namespace.yaml

# Apply deployment, service, ingress
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# Step 4: Restart deployment to pull new image
echo ""
echo "🔄 Step 4: Rolling out new deployment..."
kubectl rollout restart deployment/sentinel-poc -n sentinel-poc

# Step 5: Wait for rollout
echo ""
echo "⏳ Step 5: Waiting for rollout to complete..."
kubectl rollout status deployment/sentinel-poc -n sentinel-poc --timeout=120s

# Show status
echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 Pod status:"
kubectl get pods -n sentinel-poc

echo ""
echo "🌐 Access URL: http://sentinel-poc.fieldintech.com"
echo ""

