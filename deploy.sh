#!/bin/bash
set -e

# Configuration
AWS_ACCOUNT_ID="569061878514"
AWS_REGION="eu-west-1"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_NAME="sentinel-poc"
TAG="${1:-latest}"
NAMESPACE="sentinel-poc"
PROJECT_DIR="/home/michael/fieldin/up42-sentinel-poc"

echo "🚀 Deploying Sentinel-2 POC to Kubernetes"
echo "=========================================="
echo "Image: ${ECR_REGISTRY}/${IMAGE_NAME}:${TAG}"
echo ""

cd ${PROJECT_DIR}

# Step 1: Build client (optional - can skip if no changes)
if [[ "$2" != "--skip-build" ]]; then
    echo "📦 Step 1: Building Angular client..."
    cd client
    npm install
    npm run build
    cd ..
    echo "✅ Client built"
else
    echo "⏭️  Step 1: Skipping client build (--skip-build)"
fi

# Step 2: Build Docker image
echo ""
echo "🐳 Step 2: Building Docker image..."
docker buildx build --platform linux/amd64 --load \
    -t ${ECR_REGISTRY}/${IMAGE_NAME}:${TAG} .
echo "✅ Image built"

# Step 3: Login to ECR and push
echo ""
echo "📤 Step 3: Pushing to ECR..."
aws ecr get-login-password --region ${AWS_REGION} | \
    docker login --username AWS --password-stdin ${ECR_REGISTRY}
docker push ${ECR_REGISTRY}/${IMAGE_NAME}:${TAG}
echo "✅ Image pushed"

# Step 4: Apply Kubernetes manifests
echo ""
echo "🎯 Step 4: Applying Kubernetes manifests..."
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/service-public.yaml 2>/dev/null || true
kubectl apply -f k8s/ingress.yaml
echo "✅ Manifests applied"

# Step 5: Restart deployment
echo ""
echo "🔄 Step 5: Rolling out new deployment..."
kubectl rollout restart deployment/${IMAGE_NAME} -n ${NAMESPACE}

# Step 6: Wait for rollout
echo ""
echo "⏳ Step 6: Waiting for rollout to complete..."
kubectl rollout status deployment/${IMAGE_NAME} -n ${NAMESPACE} --timeout=120s

# Step 7: Wait for ALB health check
echo ""
echo "⏳ Step 7: Waiting for ALB health check (30s)..."
sleep 30

# Step 8: Verify deployment
echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 Pod status:"
kubectl get pods -n ${NAMESPACE}

echo ""
echo "🔗 Services:"
kubectl get svc -n ${NAMESPACE}

echo ""
echo "🌐 Public URL: http://sentinel-poc.dev.fieldintech.com"
echo ""

# Health check
echo "🩺 Health check:"
curl -s --max-time 10 http://sentinel-poc.dev.fieldintech.com/health || echo "⚠️  Health check failed - ALB may still be registering targets"
echo ""
