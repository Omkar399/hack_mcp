#!/bin/bash

# Reliable Screen Memory Capture
# Handles Chrome focus issues and ensures stable operation

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

# Function to show immediate feedback
show_immediate_feedback() {
    osascript -e 'display alert "📸 Capturing..." message "Screenshot taken" as informational giving up after 0.8' > /dev/null 2>&1 &
}

# Function to check and start API server with retries
ensure_api_server() {
    local max_attempts=3
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:5003/health > /dev/null 2>&1; then
            return 0
        fi
        
        if [ $attempt -eq 1 ]; then
            # Start server on first attempt
            uv run uvicorn screen_api:app --host 0.0.0.0 --port 5003 > api_server.log 2>&1 &
            
            # Wait for server to be ready
            for i in {1..20}; do
                if curl -s http://localhost:5003/health > /dev/null 2>&1; then
                    return 0
                fi
                sleep 0.5
            done
        fi
        
        attempt=$((attempt + 1))
        sleep 1
    done
    
    # Failed to start server
    osascript -e 'display alert "Screen Memory Error" message "Failed to start server" as critical giving up after 3'
    return 1
}

# Main execution
main() {
    # Show immediate feedback first (non-blocking)
    show_immediate_feedback
    
    # Ensure API server is running
    if ! ensure_api_server; then
        exit 1
    fi
    
    # Determine capture type from argument
    CAPTURE_TYPE=""
    if [ "$1" = "--vision" ] || [ "$1" = "-v" ]; then
        CAPTURE_TYPE="--vision"
    fi
    
    # Run the capture (suppress console output for shortcuts)
    uv run python simple_capture.py $CAPTURE_TYPE > /dev/null 2>&1 &
    
    # Don't wait for completion - let it run in background
    exit 0
}

# Run main function
main "$@" 