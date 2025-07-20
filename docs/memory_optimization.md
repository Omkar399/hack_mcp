# Memory Optimization Guide for Eidolon

## 🚨 Common Issues and Solutions

### Issue: High Memory Usage / System Freezing

**Symptoms:**
- System becomes slow or unresponsive
- Out of memory errors
- Process killed by system

**Solutions:**

#### For 8GB Systems:
```bash
# Use low memory mode
python -m eidolon start --low-memory

# Or set explicit memory limit
python -m eidolon start --memory-limit 6.0
```

#### For 16GB Systems:
```bash
# Use balanced settings
python -m eidolon start --memory-limit 12.0
```

### Issue: "Cannot load model" Errors

**Symptoms:**
- Vision model fails to load
- "Insufficient memory" warnings

**Solutions:**

1. **Close other applications** before starting Eidolon
2. **Use CPU-only mode** for vision models:
   ```bash
   export EIDOLON_USE_CPU_ONLY=1
   python -m eidolon start
   ```
3. **Disable local vision analysis** (use cloud only):
   ```bash
   export EIDOLON_DISABLE_LOCAL_VISION=1
   python -m eidolon start
   ```

### Issue: Slow Performance

**Symptoms:**
- Long delays in screenshot processing
- High CPU usage

**Solutions:**

1. **Increase capture interval**:
   ```bash
   python -m eidolon start --interval 60  # 1 minute intervals
   ```

2. **Reduce concurrent processing**:
   ```bash
   export EIDOLON_MAX_CONCURRENT=1
   python -m eidolon start
   ```

## 🛠️ Memory Configuration Options

### Environment Variables

```bash
# Memory limits
export EIDOLON_MAX_RAM_GB=8.0
export EIDOLON_USE_CPU_ONLY=1
export EIDOLON_DISABLE_LOCAL_VISION=1

# Performance tuning
export EIDOLON_MAX_CONCURRENT=1
export EIDOLON_BATCH_SIZE=1
export EIDOLON_CLEANUP_INTERVAL=300  # 5 minutes

# PyTorch optimizations
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
```

### Command Line Options

```bash
# Basic usage for low-end systems
python -m eidolon start --low-memory --interval 60 --background

# Medium systems
python -m eidolon start --memory-limit 12.0 --interval 30

# High-end systems
python -m eidolon start --memory-limit 24.0 --interval 10
```

## 📊 Memory Monitoring

### Check Current Usage
```bash
python -c "
from eidolon.utils.memory_optimizer import get_memory_optimizer
optimizer = get_memory_optimizer()
stats = optimizer.get_memory_usage()
print(f'RAM Usage: {stats[\"used_gb\"]:.1f}GB / {stats[\"total_gb\"]:.1f}GB ({stats[\"percent\"]:.1f}%)')
print(f'Available: {stats[\"available_gb\"]:.1f}GB')
print(f'Process: {stats[\"process_mb\"]:.1f}MB')
"
```

### Real-time Monitoring
```bash
# Monitor system resources while running
htop  # or top on basic systems
```

## 🎯 Recommended Settings by System

### 8GB RAM Systems
```yaml
memory_limit: 6.0
capture_interval: 60
max_concurrent: 1
use_cpu_only: true
disable_local_vision: false  # Use base model
vision_model_size: base
enable_model_unloading: true
```

### 16GB RAM Systems  
```yaml
memory_limit: 12.0
capture_interval: 30
max_concurrent: 2
use_cpu_only: false
vision_model_size: base
enable_model_unloading: false
```

### 32GB+ RAM Systems
```yaml
memory_limit: 24.0
capture_interval: 10
max_concurrent: 3
use_cpu_only: false
vision_model_size: large
enable_model_unloading: false
```

## 🔧 Troubleshooting Steps

### 1. Check System Resources
```bash
# Check total RAM
free -h  # Linux
vm_stat | grep "Pages" | awk '{print $3}' | sed 's/\.//' # macOS

# Check available space
df -h

# Check running processes
ps aux | grep eidolon
```

### 2. Clean Restart
```bash
# Stop all Eidolon processes
python -m eidolon stop
pkill -f eidolon

# Clear cache
rm -rf ~/.cache/huggingface/
rm -rf ~/.cache/torch/

# Restart with low memory mode
python -m eidolon start --low-memory
```

### 3. Reduce Data Retention
```bash
# Clean old data
python -m eidolon cleanup --days 7

# Limit screenshot retention
export EIDOLON_MAX_SCREENSHOTS=100
```

## ⚡ Performance Optimizations

### For macOS with Apple Silicon
```bash
# Enable Metal Performance Shaders
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

### For Intel/AMD Systems
```bash
# Optimize CPU usage
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
```

### For Systems with Dedicated GPU
```bash
# Use GPU for vision models
export EIDOLON_USE_GPU=1
export CUDA_VISIBLE_DEVICES=0
```

## 🆘 Emergency Recovery

If Eidolon causes system instability:

1. **Force stop all processes**:
   ```bash
   sudo pkill -9 -f eidolon
   sudo pkill -9 -f python
   ```

2. **Clear all caches**:
   ```bash
   rm -rf ~/.cache/eidolon/
   rm -rf ~/.cache/huggingface/
   rm -rf ~/.cache/torch/
   ```

3. **Restart with minimal settings**:
   ```bash
   export EIDOLON_USE_CPU_ONLY=1
   export EIDOLON_DISABLE_LOCAL_VISION=1
   export EIDOLON_MAX_RAM_GB=4.0
   python -m eidolon start --low-memory --interval 120
   ```

## 📈 Memory Usage Expectations

| Component | Typical RAM Usage |
|-----------|-------------------|
| Base System | 100-200 MB |
| Vision Model (Base) | 1-2 GB |
| Vision Model (Large) | 3-4 GB |
| Database + Vector Store | 200-500 MB |
| Screenshot Buffer | 50-200 MB |
| **Total (Conservative)** | **2-3 GB** |
| **Total (Full Features)** | **4-7 GB** |

## 🔍 Getting Help

If you continue to experience memory issues:

1. **Check logs**: `tail -f data/logs/eidolon.log`
2. **Run health check**: `./scripts/health_check.sh`
3. **Report issue** with system specs and error logs