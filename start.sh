#!/bin/bash

# Screen Memory Assistant - Startup Script
# Launches the full containerized stack

set -e

echo "🚀 Starting Screen Memory Assistant..."
echo "========================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install Docker Compose."
    exit 1
fi

# Function to wait for service
wait_for_service() {
    local service_name=$1
    local check_url=$2
    local max_attempts=30
    local attempt=1
    
    echo "⏳ Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s $check_url > /dev/null 2>&1; then
            echo "✅ $service_name is ready!"
            return 0
        fi
        
        echo "   Attempt $attempt/$max_attempts - waiting..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "❌ $service_name failed to start within timeout"
    return 1
}

# Create directories
echo "📁 Creating directories..."
mkdir -p screenshots logs database

# Start the services
echo "🐳 Starting Docker services..."
docker-compose up -d

# Wait for Postgres to be ready
echo ""
wait_for_service "PostgreSQL" "localhost:5432" || {
    echo "❌ PostgreSQL failed to start"
    echo "🔍 Checking logs..."
    docker-compose logs postgres
    exit 1
}

# Check for OpenRouter API key
if [[ -z "${OPENROUTER_API_KEY}" ]]; then
    echo ""
    echo "⚠️  No OPENROUTER_API_KEY found - vision fallback disabled"
    echo "   Set your OpenRouter API key to enable GPT-4o Vision fallback:"
    echo "   export OPENROUTER_API_KEY=your-key-here"
else
    echo ""
    echo "✅ OpenRouter API key found - vision fallback enabled"
fi

# Wait for the main app to be ready
echo ""
wait_for_service "Screen Memory API" "http://localhost:5003/health" || {
    echo "❌ Screen Memory API failed to start"
    echo "🔍 Checking logs..."
    docker-compose logs app
    exit 1
}

echo ""
echo "🎉 Screen Memory Assistant is running!"
echo "======================================="
echo ""
echo "🌐 API Server:       http://localhost:5003"
echo "🗄️  Database:        localhost:5432 (user: hack, db: screenmemory)"
echo "🤖 Vision API:       OpenRouter (GPT-4o-mini)"
echo ""
echo "📋 Quick Start Commands:"
echo "  uv run python cli.py health              # Check system status"
echo "  uv run python cli.py capture             # Take a screenshot now"
echo "  uv run python cli.py search 'docker'     # Search for text"
echo "  uv run python cli.py recent              # Show recent events"
echo ""
echo "🧪 To run tests:"
echo "  uv run python test_integration.py"
echo ""
echo "🛑 To stop:"
echo "  docker-compose down"
echo ""

# Optionally run a quick health check
if [[ "$1" == "--test" ]]; then
    echo "🧪 Running quick health check..."
    sleep 2
    python test_integration.py
fi

# Keep the script running to show logs (optional)
if [[ "$1" == "--logs" ]]; then
    echo "📋 Showing logs (Ctrl+C to exit)..."
    docker-compose logs -f
fi 