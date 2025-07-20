# Installation Guide

This guide will help you install and set up Eidolon AI Personal Assistant on your system.

## Prerequisites

- **Python 3.9+** (Python 3.11+ recommended)
- **Git** for cloning the repository
- **4GB+ RAM** for optimal performance
- **2GB+ free disk space** for data storage

### macOS Requirements

- **Screen Recording Permission**: Eidolon needs permission to capture screenshots
- **Homebrew** (optional, for installing system dependencies)

### Linux Requirements

- **X11 or Wayland** display server
- **Screenshot utilities** (usually pre-installed)

## Installation Methods

### Method 1: Automated Installation (Recommended)

The easiest way to get started:

```bash
# Clone the repository
git clone <repository-url>
cd eidolon

# Run the automated installation script
./scripts/install.sh
```

This script will:
- Create a Python virtual environment
- Install all required dependencies
- Set up data directories
- Create configuration files
- Run installation tests

### Method 2: Manual Installation

For more control over the installation process:

#### Step 1: Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd eidolon

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

#### Step 2: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install Eidolon dependencies
pip install -r requirements.txt
```

#### Step 3: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (optional)
nano .env  # or your preferred editor
```

#### Step 4: Create Data Directories

```bash
# Create required directories
mkdir -p data/{screenshots,database,logs}
```

#### Step 5: Verify Installation

```bash
# Test the installation
python -c "import eidolon; print('✅ Installation successful!')"

# Run health check
./scripts/health_check.sh
```

## Configuration

### Environment Variables

Edit `.env` to customize your installation:

```bash
# Cloud AI API Keys (optional)
GEMINI_API_KEY=your_gemini_key_here
ANTHROPIC_API_KEY=your_claude_key_here
OPENAI_API_KEY=your_openai_key_here

# System Settings
EIDOLON_DATA_DIR=./data
LOG_LEVEL=INFO
CAPTURE_INTERVAL=30
ACTIVITY_THRESHOLD=0.1
MAX_STORAGE_GB=10

# Privacy Settings
DATA_RETENTION_DAYS=90
AUTO_REDACT_SENSITIVE=true
MAX_CPU_PERCENT=80
MAX_MEMORY_MB=2048
```

### API Keys (Optional)

Eidolon works entirely locally, but you can enhance it with cloud AI:

1. **Google Gemini**: Get API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. **Anthropic Claude**: Get API key from [Anthropic Console](https://console.anthropic.com/)
3. **OpenAI**: Get API key from [OpenAI Platform](https://platform.openai.com/api-keys)

### System Permissions

#### macOS

Grant screen recording permission:

1. Go to **System Preferences** → **Security & Privacy** → **Privacy**
2. Select **Screen Recording** from the left sidebar
3. Add your terminal application or Python executable
4. Restart the terminal

#### Linux

Ensure your user has access to the display:

```bash
# Check display access
echo $DISPLAY

# If needed, grant access
xhost +local:
```

## Verification

### Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Quick system test
python -m eidolon status
```

### Health Check

```bash
# Comprehensive health check
./scripts/health_check.sh
```

Expected output:
```
✅ Python 3: Available
✅ Virtual Environment: Found
✅ Dependencies: All installed
✅ Data directories: Created
✅ Eidolon: Responding
```

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### Permission Errors (macOS)
- Grant screen recording permission in System Preferences
- Restart terminal after granting permissions

#### Memory Issues
- Reduce `MAX_MEMORY_MB` in `.env`
- Close other applications
- Consider upgrading RAM

#### Storage Issues
- Reduce `DATA_RETENTION_DAYS` in `.env`
- Run cleanup: `python -m eidolon cleanup --days 7`
- Check available disk space

### Getting Help

1. Check the [health check script](../scripts/health_check.sh)
2. Review logs in `data/logs/`
3. Run tests to identify specific issues
4. Check system requirements

## Next Steps

After successful installation:

1. **Start the system**: `./scripts/start.sh`
2. **Try a search**: `python -m eidolon search "test"`
3. **Start chatting**: `python -m eidolon chat`
4. **Check status**: `python -m eidolon status`

See the [Usage Guide](usage.md) for detailed usage instructions.
