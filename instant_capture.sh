#!/bin/bash

# Ultra-Simple Instant Capture for Chrome/Browser compatibility
# Minimal script that always works

export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
cd "$(dirname "$0")"

# No notification - silent capture (you know it works when you press the key)
# osascript -e 'beep 1' &

# Load environment
[ -f .env ] && export $(cat .env | grep -v '^#' | xargs)

# Quick server check and start if needed
if ! curl -s http://localhost:5003/health >/dev/null 2>&1; then
    uv run uvicorn screen_api:app --host 0.0.0.0 --port 5003 >/dev/null 2>&1 &
    sleep 2
fi

# Capture in background (silent to avoid duplicate notifications)
uv run python simple_capture.py --vision --silent >/dev/null 2>&1 &

exit 0 