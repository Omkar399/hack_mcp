#!/bin/bash

# Screen Memory Assistant - Start with Hotkeys
# Launches API server and hotkey daemon together

set -e

echo "🚀 Starting Screen Memory Assistant with Hotkeys..."
echo "======================================================="

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ Environment variables loaded from .env"
else
    echo "⚠️  No .env file found - using defaults"
fi

# Check if API server is already running
if curl -s http://localhost:5003/health > /dev/null 2>&1; then
    echo "✅ API server already running"
else
    echo "🚀 Starting API server..."
    
    # Start API server in background
    uv run uvicorn screen_api:app --host 0.0.0.0 --port 5003 > api_server.log 2>&1 &
    API_PID=$!
    
    # Wait for API server to be ready
    echo "⏳ Waiting for API server to start..."
    for i in {1..30}; do
        if curl -s http://localhost:5003/health > /dev/null 2>&1; then
            echo "✅ API server ready!"
            break
        fi
        sleep 1
        if [ $i -eq 30 ]; then
            echo "❌ API server failed to start within 30 seconds"
            echo "📋 Check api_server.log for details"
            exit 1
        fi
    done
fi

echo ""
echo "🔥 Starting Hotkey Daemon..."
echo "🎯 Hotkeys:"
echo "   📸 Cmd+Shift+S: Normal screenshot"
echo "   🤖 Cmd+Shift+V: Screenshot with AI analysis"
echo "   🛑 Ctrl+C: Stop daemon"
echo ""

# Start hotkey daemon (this will run in foreground)
uv run python hotkey_daemon.py

# Cleanup function
cleanup() {
    echo ""
    echo "🧹 Cleaning up..."
    if [ ! -z "$API_PID" ]; then
        kill $API_PID 2>/dev/null || true
        echo "🛑 API server stopped"
    fi
    exit 0
}

# Set up cleanup on script exit
trap cleanup EXIT INT TERM 