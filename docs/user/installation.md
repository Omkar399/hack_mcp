# Installation Guide

This guide will help you install Eidolon AI Personal Assistant on your system. Eidolon supports Windows, macOS, and Linux platforms.

## 📋 Prerequisites

Before installing Eidolon, ensure your system meets the following requirements:

### System Requirements
- **Python 3.9 or higher** (Python 3.11+ recommended)
- **4GB RAM minimum** (8GB+ recommended)
- **2GB free disk space** (10GB+ recommended for extensive usage)
- **Internet connection** for cloud AI features and model downloads

### Platform-Specific Requirements

#### Windows (10/11)
- Windows 10 version 1903 or later
- Visual C++ Redistributable (usually included with Python)
- PowerShell 5.1 or later

#### macOS (10.15+)
- macOS Catalina (10.15) or later
- Xcode Command Line Tools
- Homebrew (recommended for Python management)

#### Linux (Ubuntu 20.04+)
- Ubuntu 20.04 LTS or equivalent distribution
- `python3-dev` and `python3-pip` packages
- System dependencies for screenshot capture

## 🚀 Quick Installation

### Option 1: Install from PyPI (Recommended)

```bash
# Install Eidolon
pip install eidolon-ai

# Verify installation
python -m eidolon --version
```

### Option 2: Install from Source

```bash
# Clone the repository
git clone https://github.com/eidolon-ai/eidolon.git
cd eidolon

# Install in development mode
pip install -e .

# Verify installation
python -m eidolon --version
```

## 🔧 Detailed Installation Steps

### Step 1: Install Python

#### Windows
1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run the installer and **check "Add Python to PATH"**
3. Verify installation:
   ```cmd
   python --version
   pip --version
   ```

#### macOS
Using Homebrew (recommended):
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11

# Verify installation
python3 --version
pip3 --version
```

#### Linux (Ubuntu/Debian)
```bash
# Update package list
sudo apt update

# Install Python and pip
sudo apt install python3 python3-pip python3-dev python3-venv

# Install system dependencies
sudo apt install tesseract-ocr libtesseract-dev
sudo apt install build-essential libssl-dev libffi-dev

# Verify installation
python3 --version
pip3 --version
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv eidolon-env

# Activate virtual environment
# On Windows:
eidolon-env\Scripts\activate

# On macOS/Linux:
source eidolon-env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 3: Install Eidolon

#### From PyPI
```bash
pip install eidolon-ai
```

#### From Source (Development)
```bash
# Clone repository
git clone https://github.com/eidolon-ai/eidolon.git
cd eidolon

# Install dependencies
pip install -r requirements.txt

# Install in editable mode
pip install -e .
```

### Step 4: Install System Dependencies

#### Windows Additional Setup
```powershell
# Install Tesseract OCR
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH or set TESSERACT_CMD environment variable
```

#### macOS Additional Setup
```bash
# Install Tesseract OCR
brew install tesseract

# Install additional language packs (optional)
brew install tesseract-lang
```

#### Linux Additional Setup
```bash
# Install additional Tesseract language packs
sudo apt install tesseract-ocr-eng tesseract-ocr-osd

# Install optional dependencies for better performance
sudo apt install python3-opencv python3-numpy
```

### Step 5: Verify Installation

```bash
# Check Eidolon version
python -m eidolon --version

# Check system status
python -m eidolon status

# Run basic health check
python -m eidolon --help
```

## ⚙️ Initial Configuration

### Step 1: Create Configuration Directory

```bash
# Eidolon will create this automatically, but you can pre-create it
mkdir -p ~/.eidolon
```

### Step 2: Initialize Eidolon

```bash
# Initialize with default configuration
python -m eidolon init

# Initialize with guided setup (recommended for new users)
python -m eidolon init --guided
```

### Step 3: Configure API Keys (Optional)

For cloud AI features, configure your API keys:

```bash
# Set environment variables
export OPENAI_API_KEY="your-openai-key"
export CLAUDE_API_KEY="your-claude-key"
export GEMINI_API_KEY="your-gemini-key"

# Or edit configuration file
python -m eidolon config set analysis.cloud_apis.openai_key "your-openai-key"
```

### Step 4: Test Basic Functionality

```bash
# Start monitoring (will capture screenshots)
python -m eidolon capture --test

# Perform a test search
python -m eidolon search "test query"

# Check system status
python -m eidolon status
```

## 🐳 Docker Installation

For containerized deployment:

```bash
# Pull the Docker image
docker pull eidolon-ai/eidolon:latest

# Run Eidolon container
docker run -d \
  --name eidolon \
  -v ~/.eidolon:/app/data \
  -e OPENAI_API_KEY="your-key" \
  eidolon-ai/eidolon:latest

# Access the container
docker exec -it eidolon bash
```

## 🔍 Troubleshooting Installation

### Common Issues

#### Permission Errors
```bash
# On macOS/Linux, use --user flag
pip install --user eidolon-ai

# Or fix permissions
sudo chown -R $(whoami) ~/.local
```

#### Python Version Issues
```bash
# Check Python version
python --version

# Use specific Python version
python3.11 -m pip install eidolon-ai
```

#### Missing System Dependencies
```bash
# Ubuntu/Debian
sudo apt install python3-dev libffi-dev libssl-dev

# CentOS/RHEL
sudo yum install python3-devel libffi-devel openssl-devel

# macOS (missing Xcode tools)
xcode-select --install
```

#### Tesseract OCR Issues
```bash
# Find Tesseract installation
which tesseract

# Set environment variable if needed
export TESSERACT_CMD="/usr/local/bin/tesseract"

# Test Tesseract
tesseract --version
```

### Performance Issues

#### Slow Installation
```bash
# Use faster package index
pip install -i https://pypi.org/simple/ eidolon-ai

# Install without cache
pip install --no-cache-dir eidolon-ai
```

#### Memory Issues During Installation
```bash
# Install with limited memory usage
pip install --no-cache-dir --prefer-binary eidolon-ai
```

## 🔄 Updating Eidolon

### Update from PyPI
```bash
# Update to latest version
pip install --upgrade eidolon-ai

# Update to specific version
pip install eidolon-ai==1.2.0
```

### Update from Source
```bash
cd eidolon
git pull origin main
pip install -e .
```

## 🗑️ Uninstalling Eidolon

### Remove Package
```bash
# Uninstall Eidolon
pip uninstall eidolon-ai

# Remove configuration and data (optional)
rm -rf ~/.eidolon
```

### Clean Virtual Environment
```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf eidolon-env
```

## 🚀 Next Steps

After successful installation:

1. **Complete Initial Setup**: Follow the [Quick Start Guide](quick-start.md)
2. **Configure Settings**: Review [Basic Configuration](configuration/basic.md)
3. **Set Privacy Controls**: Configure [Privacy Settings](configuration/privacy.md)
4. **Explore Features**: Try [Core Features](features/)

## 📞 Getting Help

If you encounter issues during installation:

1. Check the [Troubleshooting Guide](troubleshooting/common-issues.md)
2. Search existing [GitHub Issues](https://github.com/eidolon-ai/eidolon/issues)
3. Create a new issue with:
   - Your operating system and version
   - Python version (`python --version`)
   - Complete error message
   - Installation method used

## 🔐 Security Considerations

### Installation Security
- Always install from trusted sources (PyPI or official GitHub)
- Verify package signatures when available
- Use virtual environments to isolate dependencies
- Keep Python and pip updated

### Post-Installation Security
- Review and configure [Privacy Settings](configuration/privacy.md)
- Set up proper API key management
- Configure data retention policies
- Enable encryption for sensitive data

---

**Installation Complete!** 🎉

Your Eidolon AI Personal Assistant is now ready to use. Continue with the [Quick Start Guide](quick-start.md) to begin your first session.