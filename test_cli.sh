#!/bin/bash
# Test script for GLM-ASR CLI

set -e

echo "=== GLM-ASR CLI Test Script ==="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if container is running
echo "1. Checking if container is running..."
if docker compose ps | grep -q "glm-asr.*Up"; then
    echo -e "${GREEN}✓${NC} Container is running"
else
    echo -e "${RED}✗${NC} Container is not running. Starting it..."
    docker compose up -d
    echo "Waiting for container to be healthy..."
    sleep 10
fi

echo ""
echo "2. Testing CLI help command..."
docker compose exec glm-asr python glm_asr_cli.py --help

echo ""
echo "3. Testing server health check..."
docker compose exec glm-asr python glm_asr_cli.py health

echo ""
echo "4. Testing transcribe command help..."
docker compose exec glm-asr python glm_asr_cli.py transcribe --help

echo ""
echo "5. Creating test data directory..."
mkdir -p data

echo ""
echo "=== CLI Tests Complete ==="
echo ""
echo "To test with actual audio files:"
echo "1. Place audio files in ./data directory"
echo "2. Run: make transcribe INPUT=/app/data/your-file.mp3"
echo "3. Or: docker compose exec glm-asr python glm_asr_cli.py transcribe /app/data/your-file.mp3"
