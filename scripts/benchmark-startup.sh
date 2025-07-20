#!/bin/bash
# Eidolon Startup Performance Benchmark

echo "🔬 Eidolon Startup Performance Benchmark"
echo "========================================"
echo ""

# Test different configurations
configs=(
    "ORIGINAL:2:4:start.sh"
    "BALANCED:4:8:fast-start.sh" 
    "PERFORMANCE:6:12:fast-start.sh"
    "BEAST_MODE:8:16:fast-start.sh"
)

results_file="benchmark_results_$(date +%Y%m%d_%H%M%S).txt"

echo "📊 Results will be saved to: $results_file"
echo ""

for config in "${configs[@]}"; do
    IFS=':' read -r name models batch script <<< "$config"
    
    echo "🧪 Testing $name configuration..."
    echo "   • Models: $models instances"
    echo "   • Batch size: $batch"
    echo "   • Script: $script"
    
    # Set environment variables
    export EIDOLON_MODEL_INSTANCES=$models
    export EIDOLON_BATCH_SIZE=$batch
    
    # Measure startup time
    start_time=$(date +%s.%N)
    
    # Start in background and measure time to first capture
    ./scripts/$script > /tmp/eidolon_test_$name.log 2>&1 &
    pid=$!
    
    # Wait for first screenshot capture
    echo "   ⏱️  Measuring startup time..."
    
    capture_detected=false
    timeout=60  # 60 second timeout
    elapsed=0
    
    while [ $elapsed -lt $timeout ]; do
        if python -m eidolon status --format json 2>/dev/null | grep -q '"capture_count": [1-9]'; then
            end_time=$(date +%s.%N)
            startup_time=$(echo "$end_time - $start_time" | bc)
            echo "   ✅ First capture in: ${startup_time}s"
            capture_detected=true
            break
        fi
        sleep 0.5
        elapsed=$((elapsed + 1))
    done
    
    if [ "$capture_detected" = false ]; then
        echo "   ❌ Timeout - startup took longer than ${timeout}s"
        startup_time="TIMEOUT"
    fi
    
    # Stop the process
    kill $pid 2>/dev/null
    python -m eidolon stop 2>/dev/null
    sleep 2
    
    # Record results
    echo "$name,$models,$batch,$startup_time" >> $results_file
    
    echo "   📝 Result recorded"
    echo ""
    
    # Brief pause between tests
    sleep 3
done

echo "🏁 Benchmark Complete!"
echo ""
echo "📊 Final Results:"
echo "=================="
cat $results_file | column -t -s','
echo ""
echo "💾 Detailed results saved to: $results_file"

# Generate performance report
echo ""
echo "🎯 Performance Analysis:"
echo "========================"

best_time=$(grep -v "TIMEOUT" $results_file | cut -d',' -f4 | sort -n | head -1)
best_config=$(grep "$best_time" $results_file | cut -d',' -f1)

echo "🥇 Fastest configuration: $best_config ($best_time seconds)"
echo ""
echo "💡 Recommendations:"
if (( $(echo "$best_time < 30" | bc -l) )); then
    echo "   ✅ Excellent performance - startup under 30 seconds"
elif (( $(echo "$best_time < 60" | bc -l) )); then
    echo "   ⚠️  Good performance - consider GPU optimizations"
else
    echo "   🔴 Slow startup - check system resources and GPU availability"
fi

echo ""
echo "🚀 To use the best configuration:"
echo "export EIDOLON_MODEL_INSTANCES=$(grep "$best_time" $results_file | cut -d',' -f2)"
echo "export EIDOLON_BATCH_SIZE=$(grep "$best_time" $results_file | cut -d',' -f3)"