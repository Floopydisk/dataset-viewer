#!/bin/bash
# Post-deployment validation script
# Run this AFTER docker-compose up -d completes and services are healthy

set -e

TIMEOUT=60
INTERVAL=5
CURL_CONNECT_TIMEOUT=5
CURL_MAX_TIME=15

echo "🔧 Starting post-deployment validation..."
echo

# Function to wait for service
wait_for_service() {
    local name=$1
    local port=$2
    local url=$3
    local elapsed=0
    
    echo "⏳ Waiting for $name (http://localhost:$port$url)..."
    while [ $elapsed -lt $TIMEOUT ]; do
        if curl -sf --connect-timeout "$CURL_CONNECT_TIMEOUT" --max-time "$CURL_MAX_TIME" "http://localhost:$port$url" > /dev/null 2>&1; then
            echo "✅ $name is healthy"
            return 0
        fi
        sleep $INTERVAL
        elapsed=$((elapsed + $INTERVAL))
    done
    echo "❌ $name did not respond within ${TIMEOUT}s"
    return 1
}

json_get_string() {
    local key=$1
    local json=$2

    if command -v jq >/dev/null 2>&1; then
        echo "$json" | jq -r --arg k "$key" '.[$k] // empty'
        return 0
    fi

    # Fallback parser for simple flat JSON payloads.
    echo "$json" | sed -n "s/.*\"$key\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/p"
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
if curl -sf --connect-timeout "$CURL_CONNECT_TIMEOUT" --max-time "$CURL_MAX_TIME" "http://localhost/healthcheck" > /dev/null 2>&1; then
    echo "✅ Reverse proxy is responding"
else
    echo "❌ Reverse proxy not responding - check nginx config"
fi

echo
echo "Testing API endpoints..."

# Test with a real dataset
DATASET="ibm/duorc"
echo "Testing /splits endpoint with $DATASET..."
if curl -sf --connect-timeout "$CURL_CONNECT_TIMEOUT" --max-time "$CURL_MAX_TIME" "http://localhost/splits?dataset=$DATASET" | grep -q "splits"; then
    echo "✅ /splits endpoint working"
else
    echo "⚠️  /splits endpoint returned empty - datasets might be queued for processing"
fi

echo
echo "Testing training endpoints..."
if curl -sf --connect-timeout "$CURL_CONNECT_TIMEOUT" --max-time "$CURL_MAX_TIME" "http://localhost/train/capabilities" | grep -q "task_types"; then
    echo "✅ /train/capabilities endpoint working"
else
    echo "❌ /train/capabilities endpoint failed"
    echo "   This usually means the API container is not running the latest image/code."
    echo "   Suggested fix: docker-compose -f docker-compose.ec2.yml --env-file .env.production up -d --build api reverse-proxy"
fi

if curl -sf --connect-timeout "$CURL_CONNECT_TIMEOUT" --max-time "$CURL_MAX_TIME" "http://localhost/api/train/capabilities" | grep -q "training_algorithms"; then
    echo "✅ /api/train/capabilities alias working"
else
    echo "❌ /api/train/capabilities alias failed"
    echo "   Suggested fix: restart api + reverse-proxy containers and validate again."
fi

echo "Testing direct API container train route..."
if curl -sf --connect-timeout "$CURL_CONNECT_TIMEOUT" --max-time "$CURL_MAX_TIME" "http://localhost:8080/train/capabilities" | grep -q "task_types"; then
    echo "✅ API service exposes /train/capabilities directly"
else
    echo "❌ API service on :8080 does not expose /train/capabilities"
    echo "   API container is likely stale. Rebuild with:"
    echo "   docker-compose -f docker-compose.ec2.yml --env-file .env.production up -d --build api"
fi

echo
echo "Running end-to-end training smoke test..."
SMOKE_PAYLOAD='{"dataset":"ibm/duorc","revision":"main","modelName":"bert-base-uncased","epochs":1,"batchSize":2,"learningRate":0.001,"trainSplit":"train","experimentName":"deployment-smoke-test"}'

train_response=$(curl -sS --connect-timeout "$CURL_CONNECT_TIMEOUT" --max-time "$CURL_MAX_TIME" \
    -H "Content-Type: application/json" \
    -X POST "http://localhost/train" \
    --data "$SMOKE_PAYLOAD" || true)

smoke_job_id=$(json_get_string "job_id" "$train_response")
smoke_dataset=$(json_get_string "dataset" "$train_response")

if [ -z "$smoke_job_id" ]; then
    echo "❌ Training smoke test failed to enqueue job"
    echo "   Response: $train_response"
    echo "   Check API/worker logs: docker-compose logs api worker --tail=100"
    exit 1
fi

if [ -z "$smoke_dataset" ]; then
    smoke_dataset="ibm/duorc"
fi

echo "✅ Training job enqueued (job_id=$smoke_job_id)"

SMOKE_POLL_INTERVAL=3
SMOKE_MAX_POLLS=12
smoke_attempt=1
smoke_status=""

while [ $smoke_attempt -le $SMOKE_MAX_POLLS ]; do
    status_response=$(curl -sS --connect-timeout "$CURL_CONNECT_TIMEOUT" --max-time "$CURL_MAX_TIME" \
        -G "http://localhost/train" \
        --data-urlencode "dataset=$smoke_dataset" \
        --data-urlencode "job_id=$smoke_job_id" || true)

    smoke_status=$(json_get_string "status" "$status_response")

    if [ "$smoke_status" = "succeeded" ]; then
        echo "✅ Training smoke test succeeded"
        break
    fi

    if [ "$smoke_status" = "failed" ]; then
        echo "❌ Training smoke test returned failed status"
        echo "   Response: $status_response"
        exit 1
    fi

    echo "ℹ️  Smoke poll $smoke_attempt/$SMOKE_MAX_POLLS status=${smoke_status:-unknown}"
    sleep $SMOKE_POLL_INTERVAL
    smoke_attempt=$((smoke_attempt + 1))
done

if [ "$smoke_status" != "succeeded" ]; then
    echo "❌ Training smoke test timed out before completion"
    echo "   Last status: ${smoke_status:-unknown}"
    echo "   Check worker progress: docker-compose logs worker --tail=150"
    exit 1
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
