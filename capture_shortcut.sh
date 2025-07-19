#!/bin/bash

# Screen Memory Assistant - Shortcut Helper
# Use this script in macOS Shortcuts app

set -e

# Setup PATH for Homebrew and uv
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables if .env exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check if API server is running, start if needed
if ! curl -s http://localhost:5003/health > /dev/null 2>&1; then
    echo "🚀 Starting API server..."
    uv run uvicorn screen_api:app --host 0.0.0.0 --port 5003 > api_server.log 2>&1 &
    
    # Wait for server to be ready (longer timeout)
    for i in {1..30}; do
        if curl -s http://localhost:5003/health > /dev/null 2>&1; then
            break
        fi
        sleep 1
        if [ $i -eq 30 ]; then
            # Fallback: show error notification and exit
            osascript -e 'display alert "Screen Memory Error" message "API server failed to start" as critical giving up after 3'
            exit 1
        fi
    done
fi

# Determine capture type from argument
CAPTURE_TYPE=""
if [ "$1" = "--vision" ] || [ "$1" = "-v" ]; then
    CAPTURE_TYPE="--vision"
fi

# Run the capture
uv run python simple_capture.py $CAPTURE_TYPE 