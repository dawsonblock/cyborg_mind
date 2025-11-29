#!/bin/bash
# CyborgMind V2.6 Quick Start Script
# Ensure this script is executable before running:
#   chmod +x quick_start.sh

set -e

echo "════════════════════════════════════════════════════════════════"
echo "  🧠 CyborgMind V2.6 - Quick Start"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found. Please install docker-compose first."
    exit 1
fi

echo "✓ Docker found"
echo "✓ docker-compose found"
echo ""

# Verify build
echo "📦 Running build verification..."
python3 build_verify.py
if [ $? -ne 0 ]; then
    echo "❌ Build verification failed"
    exit 1
fi
echo ""

# Build and start
echo "🐋 Building Docker images..."
docker-compose build

echo ""
echo "🚀 Starting services..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to start..."
sleep 10

# Health checks
echo "🏥 Checking service health..."

# Check CyborgMind API
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✓ CyborgMind API: http://localhost:8000"
    echo "  📖 API Docs: http://localhost:8000/docs"
else
    echo "⚠️  CyborgMind API not ready yet (may need more time)"
fi

# Check Grafana
if curl -s http://localhost:3000 > /dev/null; then
    echo "✓ Grafana: http://localhost:3000 (admin/admin)"
else
    echo "⚠️  Grafana not ready yet"
fi

# Check Prometheus
# Prometheus runs on port 9090 inside the container, but is exposed on host port 9091.
# This health check verifies the host-exposed port (9091), which is intentional.
if curl -s http://localhost:9091 > /dev/null; then
    echo "✓ Prometheus: http://localhost:9091"
else
    echo "⚠️  Prometheus not ready yet"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✨ CyborgMind V2.6 is running!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  • Open http://localhost:8000/docs for API documentation"
echo "  • Open http://localhost:3000 for Grafana dashboards"
echo "  • Run 'docker-compose logs -f cyborgmind' to view logs"
echo "  • Run 'docker-compose down' to stop services"
echo ""
