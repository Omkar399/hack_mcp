# Eidolon Installation Guide

This guide will help you install and set up Eidolon AI Personal Assistant on your system.

## Prerequisites

### System Requirements
- **Python**: 3.9 or higher
- **Operating System**: macOS 10.15+, Windows 10+, or Linux (Ubuntu 20.04+)
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 10GB free space minimum
- **Internet**: Optional (for cloud AI features)

### Required Permissions (macOS)

⚠️ **Important**: Eidolon needs screen recording permissions to capture screenshots.

1. **Grant Screen Recording Permission**:
   - Go to `System Preferences > Security & Privacy > Privacy`
   - Select `Screen Recording` in the left panel
   - Click the lock icon and enter your password
   - Check the box next to `Terminal` (or your Python IDE)
   - Restart Terminal/IDE after granting permissions

2. **Grant Accessibility Permission** (for advanced features):
   - Go to `System Preferences > Security & Privacy > Privacy`
   - Select `Accessibility` in the left panel
   - Add Terminal or your Python IDE to the list

### Optional Dependencies

#### Tesseract OCR (for text extraction)
```bash
# macOS (using Homebrew)
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows (using Chocolatey)
choco install tesseract

# Or download from: https://github.com/UB-Mannheim/tesseract/wiki
```

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/eidolon-ai/eidolon.git
cd eidolon
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install mss Pillow psutil pyyaml pydantic python-dotenv click numpy

# For development (optional)
pip install pytest pytest-cov pytest-mock black flake8 mypy
```

### 4. Set Up Configuration

```bash
# Copy environment variables template
cp .env.example .env

# Edit .env file to add your API keys (optional)
nano .env
```

### 5. Test Installation

```bash
# Test basic imports
python -c "
import sys
sys.path.insert(0, '.')
from eidolon.core.observer import Observer
print('✅ Core modules imported successfully')

observer = Observer()
print('✅ Observer created successfully')

screenshot = observer.capture_screenshot()
print('✅ Screenshot capture:', 'working' if screenshot else 'failed')
"
```

### 6. Test CLI Interface

```bash
# Test CLI commands
python -m eidolon.cli.main --help
python -m eidolon.cli.main version
python -m eidolon.cli.main status
```

## Quick Start

### Start Screenshot Monitoring

```bash
# Start monitoring (Ctrl+C to stop)
python -m eidolon.cli.main capture

# Or with custom interval
python -m eidolon.cli.main capture --interval 30
```

### Check System Status

```bash
python -m eidolon.cli.main status
```

### Search Captured Content

```bash
python -m eidolon.cli.main search "your search term"
```

## Troubleshooting

### Permission Issues (macOS)

**Problem**: Screenshot capture fails with permission error.
**Solution**: 
1. Go to System Preferences > Security & Privacy > Privacy > Screen Recording
2. Add Terminal or your Python IDE to the allowed list
3. Restart the application after granting permissions

### Import Errors

**Problem**: `ModuleNotFoundError` when running commands.
**Solution**:
1. Make sure virtual environment is activated
2. Install missing dependencies: `pip install <missing-package>`
3. Verify you're running from the project root directory

### Performance Issues

**Problem**: High CPU or memory usage.
**Solution**:
1. Increase capture interval: `--interval 60`
2. Check configuration in `config/settings.yaml`
3. Adjust resource limits in configuration

### Storage Issues

**Problem**: Running out of disk space.
**Solution**:
1. Run cleanup: `python -m eidolon.cli.main cleanup --days 7`
2. Adjust storage limits in configuration
3. Enable automatic cleanup in settings

### OCR Not Working

**Problem**: Text extraction fails.
**Solution**:
1. Install Tesseract OCR (see prerequisites)
2. Verify installation: `tesseract --version`
3. Check OCR configuration in `config/settings.yaml`

## Configuration

### Environment Variables (.env)

```bash
# Cloud AI API Keys (optional)
GEMINI_API_KEY=your_gemini_key
CLAUDE_API_KEY=your_claude_key
OPENAI_API_KEY=your_openai_key

# Performance Settings
MAX_MEMORY_MB=1000
MAX_CPU_PERCENT=10

# Privacy Settings
LOCAL_ONLY_MODE=false
DATA_RETENTION_DAYS=365
```

### Main Configuration (config/settings.yaml)

Key settings you might want to adjust:

```yaml
observer:
  capture_interval: 10          # Seconds between screenshots
  activity_threshold: 0.05      # Change detection sensitivity
  max_storage_gb: 50           # Storage limit

privacy:
  auto_redaction: true         # Hide sensitive information
  excluded_apps:               # Apps to never monitor
    - "1Password"
    - "Keychain Access"

analysis:
  routing:
    local_first: true          # Use local AI models first
    cost_limit_daily: 10.0     # Daily cloud API cost limit
```

## Advanced Setup

### Development Mode

```bash
# Install package in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov
```

### Using Docker (Future)

```bash
# Build Docker image
docker build -t eidolon .

# Run container
docker run -v $(pwd)/data:/app/data eidolon
```

## Getting Help

### Documentation
- **README.md**: Project overview and features
- **CLAUDE.md**: Technical specifications
- **PROGRESS_PLAN.md**: Development roadmap

### Support Channels
- **GitHub Issues**: [Report bugs and request features](https://github.com/eidolon-ai/eidolon/issues)
- **GitHub Discussions**: [Community discussions](https://github.com/eidolon-ai/eidolon/discussions)

### Common Commands Reference

```bash
# Start monitoring
python -m eidolon.cli.main capture

# Stop monitoring (if running in background)
python -m eidolon.cli.main stop

# Check status
python -m eidolon.cli.main status

# Search content
python -m eidolon.cli.main search "query"

# Clean old data
python -m eidolon.cli.main cleanup --days 30

# Export data
python -m eidolon.cli.main export --path backup.json

# Get help
python -m eidolon.cli.main --help
```

## Security Considerations

1. **API Keys**: Store in `.env` file, never commit to version control
2. **Screenshots**: May contain sensitive information, review privacy settings
3. **Network**: Cloud features send data to AI providers, disable if needed
4. **Permissions**: Only grant necessary system permissions
5. **Storage**: Screenshots are stored locally, ensure adequate security

## Next Steps

After successful installation:

1. **Review Privacy Settings**: Ensure settings match your privacy preferences
2. **Test Core Features**: Try capturing and searching screenshots
3. **Customize Configuration**: Adjust intervals and thresholds for your needs
4. **Set Up API Keys**: Enable cloud AI features (optional)
5. **Schedule Regular Cleanup**: Set up data retention policies

---

**Note**: Eidolon is under active development. Features and installation procedures may change. Check the latest documentation for updates.