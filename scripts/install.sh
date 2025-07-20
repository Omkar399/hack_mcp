#!/bin/bash
# Eidolon AI Personal Assistant - Installation Script
# Complete setup and installation

set -e

echo "🚀 Eidolon AI Personal Assistant - Installation"
echo "=============================================="

# Check Python version
echo "🐍 Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3.9+ and try again."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python $PYTHON_VERSION found"

# Create virtual environment
echo "🔧 Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing Eidolon dependencies..."
pip install -r requirements.txt

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/{screenshots,database,logs}
echo "✅ Data directories created"

# Create environment file
if [ ! -f ".env" ]; then
    echo "🔧 Creating environment configuration..."
    cat > .env << EOF
# Eidolon AI Personal Assistant Configuration
# Copy this file and customize as needed

# Data directory (optional)
EIDOLON_DATA_DIR=./data

# Cloud AI API Keys (optional - for enhanced features)
# GEMINI_API_KEY=your_gemini_key_here
# ANTHROPIC_API_KEY=your_claude_key_here
# OPENAI_API_KEY=your_openai_key_here

# Logging level
LOG_LEVEL=INFO

# Screenshot settings
CAPTURE_INTERVAL=30
ACTIVITY_THRESHOLD=0.1
EOF
    echo "✅ Environment file created (.env)"
    echo "💡 Edit .env to add your API keys for enhanced features"
else
    echo "✅ Environment file already exists"
fi

# Test installation
echo "🧪 Testing installation..."
if python -c "import eidolon; print('✅ Eidolon package imported successfully')" 2>/dev/null; then
    echo "✅ Installation test passed"
else
    echo "⚠️  Installation test failed - some features may not work"
fi

echo ""
echo "🎉 Installation Complete!"
echo "========================"
echo ""
echo "Quick Start:"
echo "  ./scripts/start.sh          # Start Eidolon"
echo "  python -m eidolon start     # Start manually"
echo "  python -m eidolon status    # Check status"
echo "  python -m eidolon search    # Search your data"
echo "  python -m eidolon chat      # Interactive chat"
echo ""
echo "For help: python -m eidolon --help"
