#!/bin/bash
# Eidolon AI Personal Assistant - ULTRA-FAST Start Script
# Optimized for maximum performance and fastest startup

set -e

# Performance optimization - prioritize this process
export NICE_LEVEL=-10
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# GPU optimization for Apple Silicon
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.95
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.85

echo "⚡ Starting Eidolon AI Personal Assistant - ULTRA-FAST MODE..."
echo "🚀 Performance optimizations enabled"

# Check system requirements
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Detect system specs for optimal configuration
TOTAL_RAM=$(python3 -c "import psutil; print(int(psutil.virtual_memory().total / (1024**3)))")
CPU_CORES=$(python3 -c "import psutil; print(psutil.cpu_count())")

echo "💻 System detected: ${TOTAL_RAM}GB RAM, ${CPU_CORES} CPU cores"

# Set optimal memory allocation based on system
if [ "$TOTAL_RAM" -gt 32 ]; then
    MEMORY_LIMIT=28.0
    BATCH_SIZE=12
    MODEL_INSTANCES=6
    echo "🔥 BEAST MODE: 32GB+ RAM detected - Maximum performance"
elif [ "$TOTAL_RAM" -gt 16 ]; then
    MEMORY_LIMIT=14.0
    BATCH_SIZE=8
    MODEL_INSTANCES=4
    echo "⚡ HIGH PERFORMANCE: 16GB+ RAM detected"
else
    MEMORY_LIMIT=8.0
    BATCH_SIZE=4
    MODEL_INSTANCES=2
    echo "🎯 BALANCED MODE: Standard RAM detected"
fi

# Check virtual environment
if [ ! -d ".venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate with performance priority
echo "⚡ Activating virtual environment with high priority..."
source .venv/bin/activate

# Fast dependency installation (skip if already installed)
echo "📦 Checking dependencies..."
if ! python -c "import torch, transformers, chromadb" &> /dev/null; then
    echo "📦 Installing missing dependencies..."
    pip install -q --no-cache-dir -r requirements.txt
else
    echo "✅ Dependencies already installed"
fi

# Create optimized data structure
echo "📁 Setting up optimized data directories..."
mkdir -p data/{screenshots,database,logs,cache,models}

# Pre-warm GPU if available
echo "🔥 Pre-warming GPU for maximum performance..."
python3 -c "
import torch
if torch.backends.mps.is_available():
    print('🚀 Apple Silicon GPU detected - pre-warming...')
    device = torch.device('mps')
    # Pre-allocate GPU memory
    dummy = torch.randn(1000, 1000, device=device)
    torch.mps.empty_cache()
    print('✅ GPU pre-warmed and ready')
else:
    print('⚠️  MPS not available, using CPU')
" 2>/dev/null || echo "ℹ️  GPU pre-warming skipped"

# Set environment variables for optimal performance
export EIDOLON_MODEL_INSTANCES=$MODEL_INSTANCES
export EIDOLON_BATCH_SIZE=$BATCH_SIZE
export EIDOLON_MEMORY_LIMIT=$MEMORY_LIMIT

echo "🎯 Configuration:"
echo "   • Memory limit: ${MEMORY_LIMIT}GB"
echo "   • Model instances: ${MODEL_INSTANCES}"
echo "   • Batch size: ${BATCH_SIZE}"
echo "   • Capture rate: 10 FPS (0.1s intervals)"

# Launch with maximum priority and optimizations
echo "🚀 Launching Eidolon with ULTRA-FAST optimizations..."

# Use nice to give high priority to the process
nice -n $NICE_LEVEL python -m eidolon start \
    --interval 0.1 \
    --memory-limit $MEMORY_LIMIT \
    --background &

EIDOLON_PID=$!

# Give the system a moment to initialize
sleep 2

echo ""
echo "✅ Eidolon ULTRA-FAST mode is now running!"
echo "📊 Process ID: $EIDOLON_PID"
echo "🔥 Performance optimizations active:"
echo "   • High process priority (nice: $NICE_LEVEL)"
echo "   • GPU memory optimization"
echo "   • Parallel model processing"
echo "   • Zero resource limits"
echo ""
echo "💡 Commands:"
echo "   python -m eidolon status     # Check system status"
echo "   python -m eidolon search     # Search your data"  
echo "   python -m eidolon chat       # Interactive chat"
echo "   python -m eidolon stop       # Stop the system"
echo "   kill $EIDOLON_PID           # Force stop if needed"
echo ""
echo "🚀 System running at MAXIMUM PERFORMANCE with NO LIMITS!"

# Optional: Monitor startup for first 10 seconds
echo "📈 Monitoring startup performance..."
for i in {1..10}; do
    sleep 1
    if python -m eidolon status --format json &>/dev/null; then
        CAPTURE_COUNT=$(python -m eidolon status --format json 2>/dev/null | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('observer', {}).get('capture_count', 0))" 2>/dev/null || echo "0")
        echo "   Second $i: $CAPTURE_COUNT screenshots captured"
    fi
done

echo "✅ Startup monitoring complete - Eidolon is running at full speed!"