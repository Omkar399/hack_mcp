#!/bin/bash
# Eidolon AI Personal Assistant - Start Script
# Simple script to start the Eidolon system

set -e

echo "🚀 Starting Eidolon AI Personal Assistant..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install/update dependencies
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

# Create data directories
echo "📁 Setting up data directories..."
mkdir -p data/{screenshots,database,logs}

# Start the system
echo "🎯 Starting Eidolon system..."
python -m eidolon start --background

echo "✅ Eidolon is now running in the background!"
echo "💡 Use 'python -m eidolon status' to check system status"
echo "🔍 Use 'python -m eidolon search \"your query\"' to search"
echo "💬 Use 'python -m eidolon chat' for interactive chat"
echo "⏹️  Use 'python -m eidolon stop' to stop the system"
