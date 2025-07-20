# Eidolon AI Personal Assistant - Complete User Guide

This unified guide combines installation, usage, and examples into a single comprehensive resource.

## Table of Contents

1. [Installation & Setup](#installation--setup)
2. [Quick Start Guide](#quick-start-guide)
3. [Usage Examples](#usage-examples)
4. [Configuration](#configuration)
5. [Advanced Features](#advanced-features)
6. [Troubleshooting](#troubleshooting)

---

# Installation & Setup

## System Requirements

- **Python**: 3.9 or higher
- **Operating System**: macOS 10.15+, Windows 10+, or Linux (Ubuntu 20.04+)
- **Memory**: 8GB RAM minimum (16GB recommended for AI features)
- **Storage**: 20GB free space minimum
- **Internet**: Required for initial setup and cloud AI features

## Prerequisites

### macOS Permissions

⚠️ **Important**: Eidolon needs screen recording permissions.

1. **Grant Screen Recording Permission**:
   - Go to `System Preferences > Security & Privacy > Privacy`
   - Select `Screen Recording` in the left panel
   - Check the box next to `Terminal` or your Python IDE
   - Restart Terminal/IDE after granting permissions

2. **Grant Accessibility Permission** (optional):
   - Go to `System Preferences > Security & Privacy > Privacy`
   - Select `Accessibility` in the left panel
   - Add Terminal or your Python IDE

### Install Tesseract OCR

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows
choco install tesseract
# Or download from: https://github.com/UB-Mannheim/tesseract/wiki
```

## Installation Steps

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/eidolon-ai/eidolon.git
cd eidolon

# Create virtual environment
python3 -m venv eidolon_env
source eidolon_env/bin/activate  # Windows: eidolon_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Verify installation
python validate_dependencies.py
```

### 2. Configure Environment

```bash
# Optional: Set up cloud AI API keys
export GEMINI_API_KEY="your-gemini-key"
export CLAUDE_API_KEY="your-claude-key"
export OPENROUTER_API_KEY="your-openrouter-key"  # Cost-effective Claude access
export OPENAI_API_KEY="your-openai-key"
```

**💡 OpenRouter.ai Integration:** Eidolon includes native support for [OpenRouter.ai](https://openrouter.ai/), which provides cost-effective access to Claude and other models. Often more economical than direct API access.

---

# Quick Start Guide

## Basic Workflow

```bash
# 1. Activate virtual environment
source eidolon_env/bin/activate

# 2. Start monitoring
python -m eidolon capture

# 3. Let it run while you work...

# 4. Search your activity
python -m eidolon search "python code"

# 5. Check system status
python -m eidolon status
```

## Your First Session

1. **Start Capturing**:
   ```bash
   python -m eidolon capture --interval 10
   ```

2. **Work normally** on your computer for a few minutes

3. **Stop capture** with Ctrl+C

4. **Search your activity**:
   ```bash
   python -m eidolon search "text"
   ```

---

# Usage Examples

## 📸 Screenshot Monitoring

### Basic Monitoring
```bash
# Default 10-second intervals
python -m eidolon capture

# Custom interval
python -m eidolon capture --interval 30

# Background mode
python -m eidolon capture --background
```

### Activity-Specific Monitoring
```bash
# Fast capture for coding (5 seconds)
python -m eidolon capture --interval 5

# Slower for reading (60 seconds)
python -m eidolon capture --interval 60

# Meeting mode (30 seconds)
python -m eidolon capture --interval 30
```

## 🔍 Searching Your Activity

### Basic Text Search
```bash
# Simple searches
python -m eidolon search "python"
python -m eidolon search "email"
python -m eidolon search "TODO"

# Limit results
python -m eidolon search "code" --limit 5

# JSON output
python -m eidolon search "function" --format json
```

### Natural Language Queries (Phase 4)
```bash
# Time-based queries
python -m eidolon search "What did I work on yesterday?"
python -m eidolon search "Show me Python code from this morning"
python -m eidolon search "Find debugging sessions from today"

# Activity summaries
python -m eidolon search "Summarize my programming work this week"
python -m eidolon search "What emails did I read today?"
```

### Advanced Search Filters
```bash
# By content type
python -m eidolon search "code" --content-type development
python -m eidolon search "website" --content-type browser
python -m eidolon search "document" --content-type document

# By time range
python -m eidolon search "python" --from "2024-01-01" --to "2024-01-31"

# Semantic search
python -m eidolon search "machine learning tutorials" --semantic
```

## 📊 Real-World Workflows

### Development Workflow
```bash
# 1. Start monitoring during coding
python -m eidolon capture --interval 5

# 2. Later, find specific code
python -m eidolon search "def calculate" --content-type development

# 3. Find error messages
python -m eidolon search "error" --limit 10

# 4. Summarize debugging session
python -m eidolon search "What errors did I fix today?"
```

### Research Workflow
```bash
# 1. Monitor research session
python -m eidolon capture --interval 30

# 2. Find articles
python -m eidolon search "research paper" --content-type browser

# 3. Extract key points
python -m eidolon search "summarize research on AI safety"
```

### Meeting Workflow
```bash
# 1. Capture during meeting
python -m eidolon capture --interval 15

# 2. Find meeting content
python -m eidolon search "meeting agenda"

# 3. Extract action items
python -m eidolon search "TODO" --from "10:00" --to "11:00"
```

---

# Configuration

## Configuration File

Edit `config/settings.yaml`:

```yaml
# Observer settings
observer:
  capture_interval: 10          # Seconds between captures
  activity_threshold: 0.05      # Change detection sensitivity
  storage_path: "./data/screenshots"
  max_storage_gb: 50

# Analysis settings
analysis:
  local_models:
    vision: "microsoft/florence-2-base"
    clip: "openai/clip-vit-base-patch32"
  cloud_apis:
    gemini_key: "${GEMINI_API_KEY}"
    claude_key: "${CLAUDE_API_KEY}"
    openrouter_key: "${OPENROUTER_API_KEY}"  # OpenRouter.ai integration
    openai_key: "${OPENAI_API_KEY}"
  routing:
    importance_threshold: 0.7
    cost_limit_daily: 10.0
    local_first: true

# Memory settings
memory:
  vector_db: "chromadb"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  chunk_size: 512
  db_path: "./data/memory/metadata.db"

# Privacy settings
privacy:
  local_only_mode: false
  sensitive_patterns:
    - "password"
    - "api_key"
    - "secret"
  excluded_apps:
    - "1Password"
    - "Keychain Access"
```

## Environment Variables

```bash
# Cloud AI API Keys
export GEMINI_API_KEY="your-key"
export CLAUDE_API_KEY="your-key"
export OPENROUTER_API_KEY="your-key"  # OpenRouter.ai for cost-effective Claude access
export OPENAI_API_KEY="your-key"

# Performance settings
export EIDOLON_MAX_MEMORY_MB="8192"
export EIDOLON_MAX_CPU_PERCENT="20"

# Privacy mode
export EIDOLON_LOCAL_ONLY="true"
```

---

# Advanced Features

## AI-Powered Analysis

### Local AI (Florence-2)
```python
from eidolon.core.analyzer import Analyzer

analyzer = Analyzer()
analysis = analyzer.analyze_content("screenshot.png")
print(f"Scene: {analysis.vision_analysis.scene_type}")
print(f"Objects: {analysis.vision_analysis.objects_detected}")
```

### Cloud AI Integration
```python
from eidolon.models.cloud_api import CloudAPIManager

api_manager = CloudAPIManager()
response = await api_manager.analyze_image(
    "screenshot.png",
    "What is the user working on?"
)
```

## Semantic Search

### Vector Database Queries
```python
from eidolon.storage.vector_db import VectorDatabase

vector_db = VectorDatabase()
results = vector_db.semantic_search(
    "Python web development",
    n_results=5
)
```

### Natural Language Processing
```python
from eidolon.core.memory import MemorySystem

memory = MemorySystem()
response = await memory.process_natural_language_query(
    "Summarize my coding activities from today"
)
```

## Automation Scripts

### Daily Summary Script
```python
#!/usr/bin/env python3
import asyncio
from datetime import datetime, timedelta
from eidolon.core.memory import MemorySystem

async def daily_summary():
    memory = MemorySystem()
    
    # Get today's activities
    query = f"Summarize all my activities from {datetime.now().date()}"
    response = await memory.process_natural_language_query(query)
    
    print(f"Daily Summary for {datetime.now().date()}")
    print("=" * 50)
    print(response.response)
    
    # Save to file
    with open(f"daily_summary_{datetime.now().date()}.txt", "w") as f:
        f.write(response.response)

if __name__ == "__main__":
    asyncio.run(daily_summary())
```

### Productivity Tracker
```python
#!/usr/bin/env python3
from eidolon.storage.metadata_db import MetadataDatabase
from collections import Counter
from datetime import datetime, timedelta

def analyze_productivity():
    db = MetadataDatabase()
    
    # Get last 24 hours of data
    end_time = datetime.now()
    start_time = end_time - timedelta(days=1)
    
    results = db.search_by_time_range(start_time, end_time)
    
    # Analyze content types
    content_types = Counter()
    for result in results:
        content_types[result['content_type']] += 1
    
    print("Productivity Report - Last 24 Hours")
    print("=" * 40)
    print(f"Total captures: {len(results)}")
    print("\nActivity breakdown:")
    for content_type, count in content_types.most_common():
        percentage = (count / len(results)) * 100
        print(f"  {content_type}: {count} ({percentage:.1f}%)")

if __name__ == "__main__":
    analyze_productivity()
```

---

# Troubleshooting

## Common Issues

### Permission Denied (macOS)
```bash
# Error: Can't capture screenshots
# Solution: Grant screen recording permission in System Preferences
```

### Module Import Errors
```bash
# Ensure virtual environment is activated
source eidolon_env/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### High Memory Usage
```yaml
# Reduce memory limit in settings.yaml
observer:
  max_memory_mb: 4096  # Reduce from 8192
```

### OCR Not Working
```bash
# Install Tesseract
brew install tesseract  # macOS
sudo apt-get install tesseract-ocr  # Ubuntu

# Verify installation
tesseract --version
```

### AI Models Not Loading
```bash
# Check disk space (models need ~500MB)
df -h ~/.cache/huggingface/

# Clear cache if needed
rm -rf ~/.cache/huggingface/hub/

# Install missing dependencies
pip install einops timm
```

## Performance Optimization

### Reduce CPU Usage
```yaml
# config/settings.yaml
observer:
  capture_interval: 30  # Increase interval
  max_cpu_percent: 10   # Reduce CPU limit
```

### Manage Storage
```bash
# Clean old data
python -m eidolon cleanup --days 30

# Check storage usage
python -m eidolon status --storage
```

### Optimize Search
```python
# Use specific queries
search("python function calculate")  # Good
search("code")  # Too broad

# Limit results
search("error", limit=5)

# Use time filters
search("bug", from_time="2024-01-01")
```

## Debugging

### Enable Debug Logging
```yaml
# config/logging.yaml
loggers:
  eidolon:
    level: DEBUG
```

### Check Logs
```bash
# View recent logs
tail -f logs/eidolon.log

# Search for errors
grep ERROR logs/eidolon.log
```

### Validate Installation
```bash
# Run validation script
python validate_dependencies.py

# Test individual components
python -c "from eidolon.core.observer import Observer; print('Observer OK')"
python -c "from eidolon.storage.vector_db import VectorDatabase; print('Vector DB OK')"
```

---

## 🆘 Getting Help

- **GitHub Issues**: [Report bugs](https://github.com/eidolon-ai/eidolon/issues)
- **Discussions**: [Ask questions](https://github.com/eidolon-ai/eidolon/discussions)
- **Documentation**: [Full docs](https://docs.eidolon.ai)
- **Email**: support@eidolon.ai

---

*Last updated: 2025-07-19 | Version: 0.1.0 | Phases 1-4 Complete*