#!/bin/bash
# Eidolon AI Personal Assistant - Health Check Script
# Check system status and health

echo "🔍 Eidolon System Health Check"
echo "================================"

# Check if Python is available
if command -v python3 &> /dev/null; then
    echo "✅ Python 3: Available"
else
    echo "❌ Python 3: Not found"
    exit 1
fi

# Check virtual environment
if [ -d ".venv" ]; then
    echo "✅ Virtual Environment: Found"
    source .venv/bin/activate
else
    echo "⚠️  Virtual Environment: Not found"
fi

# Check dependencies
echo "📦 Checking key dependencies..."
python3 -c "
import sys
deps = ['torch', 'transformers', 'chromadb', 'fastapi', 'click']
missing = []
for dep in deps:
    try:
        __import__(dep)
        print(f'✅ {dep}: Installed')
    except ImportError:
        print(f'❌ {dep}: Missing')
        missing.append(dep)
if missing:
    sys.exit(1)
"

# Check data directories
echo "📁 Checking data directories..."
for dir in "data" "data/screenshots" "data/database" "data/logs"; do
    if [ -d "$dir" ]; then
        echo "✅ $dir: Exists"
    else
        echo "⚠️  $dir: Missing (will be created)"
        mkdir -p "$dir"
    fi
done

# Check system status
echo "🎯 Checking Eidolon status..."
if python -m eidolon status --format json &> /dev/null; then
    echo "✅ Eidolon: Responding"
    python -m eidolon status
else
    echo "⚠️  Eidolon: Not running or not responding"
fi

echo ""
echo "🏁 Health check complete!"
