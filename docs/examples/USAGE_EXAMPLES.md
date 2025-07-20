# Eidolon Usage Examples

This guide provides practical examples of using Eidolon AI Personal Assistant for various workflows.

## 📸 Basic Screenshot Monitoring

### Start Simple Monitoring
```bash
# Activate virtual environment
source eidolon_env/bin/activate

# Start monitoring with default settings (10-second intervals)
python -m eidolon.cli.main capture

# Monitor with custom interval
python -m eidolon.cli.main capture --interval 30

# Run in background
python -m eidolon.cli.main capture --background
```

### Monitor Specific Activity
```bash
# Monitor during a coding session (fast capture)
python -m eidolon.cli.main capture --interval 5

# Monitor during reading/research (slower capture)
python -m eidolon.cli.main capture --interval 60
```

## 🔍 Searching Your Activity

### Basic Text Search
```bash
# Search for general terms
python -m eidolon.cli.main search "python code"
python -m eidolon.cli.main search "email"
python -m eidolon.cli.main search "meeting notes"

# Limit results
python -m eidolon.cli.main search "programming" --limit 5

# JSON output for scripting
python -m eidolon.cli.main search "terminal" --format json
```

### Advanced Search Filters
```bash
# Filter by content type
python -m eidolon.cli.main search "code" --content-type development
python -m eidolon.cli.main search "article" --content-type document
python -m eidolon.cli.main search "website" --content-type browser

# Search with confidence threshold
python -m eidolon.cli.main search "important" --min-confidence 0.8
```

## 🤖 AI-Enhanced Analysis

### Check AI Model Status
```bash
# Verify Florence-2 model availability
python -c "
from eidolon.core.analyzer import Analyzer
analyzer = Analyzer()
print(f'Florence-2 Vision Model: {\"✅ Available\" if analyzer._florence_available else \"❌ Not Available\"}')
print(f'Tesseract OCR: {\"✅ Available\" if analyzer._tesseract_available else \"❌ Not Available\"}')
"
```

### Analyze Specific Screenshots
```bash
# Analyze recent screenshots with AI
python -c "
from eidolon.core.analyzer import Analyzer
from pathlib import Path

analyzer = Analyzer()
screenshots = list(Path('data/screenshots').glob('*.png'))

if screenshots:
    latest = screenshots[-1]
    analysis = analyzer.analyze_content(latest)
    print(f'📄 File: {latest.name}')
    print(f'🏷️  Content Type: {analysis.content_type}')
    print(f'📝 Description: {analysis.description}')
    print(f'🎯 Confidence: {analysis.confidence:.2f}')
    print(f'🏷️  Tags: {analysis.tags}')
    
    if analysis.vision_analysis:
        va = analysis.vision_analysis
        print(f'👁️  Vision Model: {va.model_used}')
        print(f'🎬 Scene Type: {va.scene_type}')
        print(f'📸 AI Description: {va.description[:100]}...')
else:
    print('No screenshots found. Start monitoring first!')
"
```

## 📊 Database Management

### View Statistics
```bash
# Check system status
python -m eidolon.cli.main status

# Detailed database statistics
python -c "
from eidolon.storage.metadata_db import MetadataDatabase
db = MetadataDatabase()
stats = db.get_statistics()

print('📊 EIDOLON DATABASE STATISTICS')
print('=' * 40)
print(f'Total Screenshots: {stats[\"total_screenshots\"]}')
print(f'Analyzed Screenshots: {stats[\"analyzed_screenshots\"]}')
print(f'Screenshots with Text: {stats[\"screenshots_with_text\"]}')
print(f'Average Confidence: {stats.get(\"avg_confidence\", 0):.2f}')

if 'content_types' in stats:
    print('\\nContent Type Distribution:')
    for content_type, count in stats['content_types'].items():
        print(f'  {content_type}: {count}')
"
```

### Clean Up Old Data
```bash
# Remove screenshots older than 30 days
python -m eidolon.cli.main cleanup --days 30

# Remove screenshots older than 7 days
python -m eidolon.cli.main cleanup --days 7

# Export data before cleanup
python -m eidolon.cli.main export --path backup_$(date +%Y%m%d).json
```

## 🔄 Workflow Examples

### Daily Productivity Tracking
```bash
#!/bin/bash
# productivity_tracker.sh

echo "🚀 Starting Daily Productivity Tracking"

# Start monitoring
python -m eidolon.cli.main capture --interval 15 --background &
MONITOR_PID=$!

echo "📸 Monitoring started (PID: $MONITOR_PID)"
echo "💼 Work on your tasks..."
echo "⏹️  Press Enter to stop monitoring"
read

# Stop monitoring
kill $MONITOR_PID

# Generate daily report
echo "📊 Generating daily report..."
python -c "
from eidolon.storage.metadata_db import MetadataDatabase
from datetime import datetime, timedelta

db = MetadataDatabase()
end_time = datetime.now()
start_time = end_time - timedelta(days=1)

screenshots = db.get_screenshots_by_timerange(start_time, end_time)
print(f'📈 Daily Activity Summary')
print(f'   Screenshots captured: {len(screenshots)}')

# Analyze content types
content_types = {}
for shot in screenshots:
    if shot.get('content_type'):
        content_types[shot['content_type']] = content_types.get(shot['content_type'], 0) + 1

print('   Content breakdown:')
for content_type, count in sorted(content_types.items(), key=lambda x: x[1], reverse=True):
    print(f'     {content_type}: {count}')
"

echo "✅ Daily tracking complete!"
```

### Research Session Analysis
```bash
#!/bin/bash
# research_analyzer.sh

echo "🔬 Research Session Analyzer"

# Search for research-related content
echo "📚 Searching for research content..."
python -m eidolon.cli.main search "research OR study OR paper OR article" --limit 20 --format json > research_results.json

# Analyze research patterns
python -c "
import json
with open('research_results.json', 'r') as f:
    results = json.load(f)

print(f'🔍 Found {len(results)} research-related screenshots')

# Group by time of day
from datetime import datetime
import collections

hours = collections.defaultdict(int)
for result in results:
    if 'timestamp' in result:
        hour = datetime.fromisoformat(result['timestamp']).hour
        hours[hour] += 1

print('⏰ Research activity by hour:')
for hour in sorted(hours.keys()):
    bars = '█' * (hours[hour] // 2 + 1)
    print(f'   {hour:02d}:00 {bars} ({hours[hour]})')
"

rm research_results.json
echo "✅ Research analysis complete!"
```

### Development Environment Monitor
```bash
#!/bin/bash
# dev_monitor.sh

echo "💻 Development Environment Monitor"

# Start focused monitoring for development
python -m eidolon.cli.main capture --interval 10 &
MONITOR_PID=$!

echo "⌨️  Monitoring development session..."
echo "🛑 Press Enter when done coding"
read

kill $MONITOR_PID

# Analyze development session
echo "📊 Analyzing development session..."
python -c "
from eidolon.storage.metadata_db import MetadataDatabase
from datetime import datetime, timedelta

db = MetadataDatabase()
end_time = datetime.now()
start_time = end_time - timedelta(hours=1)

# Search for development-related content
dev_screenshots = db.search_text('code OR terminal OR git OR python OR javascript')

languages = {'python': 0, 'javascript': 0, 'terminal': 0, 'git': 0}
for shot in dev_screenshots:
    text = shot.get('text', '').lower()
    for lang in languages:
        if lang in text:
            languages[lang] += 1

print('💻 Development Session Summary:')
print('   Language/Tool usage:')
for tool, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
    if count > 0:
        print(f'     {tool}: {count} screenshots')
"

echo "✅ Development session analysis complete!"
```

## ⚙️ Configuration Examples

### High-Performance Setup
```bash
# Copy high-performance configuration
cp config/settings-high-performance.yaml config/settings.yaml

# Verify memory allocation
python -c "
from eidolon.utils.config import get_config
config = get_config()
memory_mb = config.observer.max_memory_mb
print(f'Memory allocation: {memory_mb}MB ({memory_mb/1024:.1f}GB)')
"
```

### Custom Content Type Patterns
```python
# Add custom patterns to analyzer.py
custom_patterns = {
    "design": [
        r"photoshop|illustrator|figma|sketch",
        r"design|layout|mockup|wireframe",
        r"color|palette|font|typography"
    ],
    "data_science": [
        r"jupyter|pandas|numpy|matplotlib",
        r"data|dataset|analysis|visualization",
        r"machine learning|AI|neural network"
    ]
}
```

## 🔧 Troubleshooting Examples

### Memory Issues
```bash
# Check system memory
python -c "
import psutil
total = psutil.virtual_memory().total / (1024**3)
available = psutil.virtual_memory().available / (1024**3)
print(f'System Memory: {available:.1f}GB available of {total:.1f}GB total')

if total < 8:
    print('⚠️  Warning: Less than 8GB RAM may limit AI model performance')
else:
    print('✅ Sufficient memory for AI models')
"

# Monitor memory usage during capture
python -c "
from eidolon.core.observer import Observer
import time, psutil

observer = Observer({'capture_interval': 5})
observer.start_monitoring()

for i in range(3):
    time.sleep(5)
    memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
    print(f'Memory usage: {memory_mb:.1f}MB')

observer.stop_monitoring()
"
```

### OCR Issues
```bash
# Test OCR functionality
python -c "
from eidolon.core.analyzer import Analyzer
analyzer = Analyzer()

print('🔍 OCR Engine Status:')
print(f'   Tesseract: {analyzer._tesseract_available}')

if not analyzer._tesseract_available:
    print('❌ Tesseract not available. Install with:')
    print('   macOS: brew install tesseract')
    print('   Ubuntu: sudo apt-get install tesseract-ocr')
"
```

## 🎯 Performance Optimization

### Optimal Capture Intervals
```python
# Different intervals for different activities
activity_intervals = {
    "coding": 5,          # Fast changes
    "reading": 30,        # Slow changes  
    "meetings": 60,       # Very slow changes
    "design_work": 10,    # Medium changes
    "research": 15        # Medium-slow changes
}
```

### Batch Operations
```bash
# Process multiple screenshots efficiently
python -c "
from eidolon.core.analyzer import Analyzer
from pathlib import Path
import time

analyzer = Analyzer()
screenshots = list(Path('data/screenshots').glob('*.png'))

print(f'📊 Batch analyzing {len(screenshots)} screenshots...')
start_time = time.time()

analyzed = 0
for screenshot in screenshots[-10:]:  # Last 10 screenshots
    try:
        analysis = analyzer.analyze_content(screenshot)
        analyzed += 1
    except Exception as e:
        print(f'❌ Failed to analyze {screenshot.name}: {e}')

elapsed = time.time() - start_time
print(f'✅ Analyzed {analyzed} screenshots in {elapsed:.1f}s')
print(f'   Average: {elapsed/max(analyzed,1):.2f}s per screenshot')
"
```

---

These examples demonstrate the power and flexibility of Eidolon for various use cases. Experiment with different configurations and workflows to find what works best for your needs! 🚀