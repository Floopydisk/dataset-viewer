#!/bin/bash
# Post-deployment validation script
# Run this AFTER docker-compose up -d completes and services are healthy

set -e

TIMEOUT=60
ELAPSED=0
INTERVAL=5

echo "🔧 Starting post-deployment validation..."
echo

# Function to wait for service
wait_for_service() {
    local name=$1
    local port=$2
    local url=$3
    
    echo "⏳ Waiting for $name (http://localhost:$port$url)..."
    while [ $ELAPSED -lt $TIMEOUT ]; do
        if curl -sf "http://localhost:$port$url" > /dev/null 2>&1; then
            echo "✅ $name is healthy"
            return 0
        fi
        sleep $INTERVAL
        ELAPSED=$((ELAPSED + $INTERVAL))
    done
    echo "❌ $name did not respond within ${TIMEOUT}s"
    return 1
}

# Check all services
echo "Checking services..."
wait_for_service "API" 8080 "/healthcheck"
wait_for_service "Rows" 8082 "/healthcheck"
wait_for_service "Search" 8083 "/healthcheck"
wait_for_service "Webhook" 8087 "/healthcheck"
wait_for_service "Admin" 8081 "/healthcheck"
wait_for_service "SSE API" 8085 "/healthcheck"

echo
echo "Testing reverse proxy..."
if curl -sf "http://localhost/healthcheck" > /dev/null 2>&1; then
    echo "✅ Reverse proxy is responding"
else
    echo "❌ Reverse proxy not responding - check nginx config"
fi

echo
echo "Testing API endpoints..."

# Test with a real dataset
DATASET="ibm/duorc"
echo "Testing /splits endpoint with $DATASET..."
if curl -sf "http://localhost/splits?dataset=$DATASET" | grep -q "splits"; then
    echo "✅ /splits endpoint working"
else
    echo "⚠️  /splits endpoint returned empty - datasets might be queued for processing"
fi

echo
echo "Testing training endpoints..."
if curl -sf "http://localhost/train/capabilities" | grep -q "task_types"; then
    echo "✅ /train/capabilities endpoint working"
else
    echo "❌ /train/capabilities endpoint failed"
    echo "   This usually means the API container is not running the latest image/code."
    echo "   Suggested fix: docker-compose -f docker-compose.ec2.yml --env-file .env.production up -d --build api reverse-proxy"
fi

if curl -sf "http://localhost/api/train/capabilities" | grep -q "training_algorithms"; then
    echo "✅ /api/train/capabilities alias working"
else
    echo "❌ /api/train/capabilities alias failed"
    echo "   Suggested fix: restart api + reverse-proxy containers and validate again."
fi

echo "Testing direct API container train route..."
if curl -sf "http://localhost:8080/train/capabilities" | grep -q "task_types"; then
    echo "✅ API service exposes /train/capabilities directly"
else
    echo "❌ API service on :8080 does not expose /train/capabilities"
    echo "   API container is likely stale. Rebuild with:"
    echo "   docker-compose -f docker-compose.ec2.yml --env-file .env.production up -d --build api"
fi

echo
echo "Checking Docker compose status..."
docker-compose ps

echo
echo "Checking storage connectivity..."
if docker-compose exec -T api curl -s "$S3_ENDPOINT_URL" > /dev/null 2>&1; then
    echo "✅ S3 endpoint accessible"
else
    echo "⚠️  S3 endpoint check inconclusive (may require credentials)"
fi

echo
echo "Checking available disk space..."
df -h /mnt/data

echo
echo "✨ Post-deployment validation complete!"
echo
echo "Next steps:"
echo "1. Check worker logs: docker-compose logs worker"
echo "2. Monitor service health: docker-compose ps"
echo "3. Test an API endpoint: curl http://localhost/splits?dataset=ibm/duorc"
