#!/bin/bash

# Screen Memory Assistant - Fixed Instant Capture
# Properly captures the foreground application

set -e

# Setup PATH for Homebrew and uv
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables if .env exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

echo "📸 Capturing foreground app..."

# Give a tiny delay to ensure proper window detection
sleep 0.1

# Use the proper foreground capture system
source .venv/bin/activate && python capture_foreground.py

# Show success notification
osascript -e 'display notification "Screenshot captured and processed" with title "📸 Screen Memory" sound name "Glass"'

echo "✅ Capture complete!" 