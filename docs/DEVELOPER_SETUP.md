# Eidolon AI Personal Assistant - Developer Setup Guide

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Development Workflow](#development-workflow)
6. [Testing](#testing)
7. [Code Quality](#code-quality)
8. [Debugging](#debugging)
9. [Contributing](#contributing)
10. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Python**: 3.9 or higher (3.13 recommended)
- **Operating System**: 
  - macOS 10.15+ (preferred)
  - Windows 10/11
  - Linux (Ubuntu 20.04+)
- **Memory**: 8GB RAM minimum (16GB recommended for AI models)
- **Storage**: 10GB free space (for screenshots and models)

### Required Software

#### macOS
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.13
brew install python@3.13

# Install Tesseract for OCR
brew install tesseract

# Install additional language packs (optional)
brew install tesseract-lang
```

#### Ubuntu/Linux
```bash
# Update package manager
sudo apt update

# Install Python 3.13 and pip
sudo apt install python3.13 python3.13-pip python3.13-venv

# Install Tesseract OCR
sudo apt install tesseract-ocr tesseract-ocr-eng

# Install system dependencies
sudo apt install build-essential libssl-dev libffi-dev
sudo apt install libjpeg-dev libpng-dev libtiff-dev
sudo apt install libx11-dev libxrandr-dev

# For video capture (optional)
sudo apt install ffmpeg
```

#### Windows
```powershell
# Install Python 3.13 from Microsoft Store or python.org
# Download and install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki

# Install Git for Windows
# Download from: https://git-scm.com/download/win

# Install Visual Studio Build Tools (for some dependencies)
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

### Optional Dependencies

#### GPU Support (NVIDIA)
```bash
# For CUDA support (AI models)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

## Environment Setup

### 1. Clone the Repository

```bash
# Clone the repository
git clone https://github.com/your-org/eidolon.git
cd eidolon

# Create and activate virtual environment
python3.13 -m venv eidolon_env
source eidolon_env/bin/activate  # macOS/Linux
# OR
eidolon_env\Scripts\activate     # Windows
```

### 2. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install development dependencies
pip install -r requirements-dev.txt

# Install project in development mode
pip install -e .

# Verify installation
python -m eidolon --help
```

### 3. Environment Variables

Create a `.env` file in the project root:

```bash
# Copy example environment file
cp .env.example .env

# Edit with your API keys
nano .env  # or your preferred editor
```

Example `.env` file:
```bash
# Cloud AI API Keys (optional but recommended)
GEMINI_API_KEY=your_gemini_api_key_here
CLAUDE_API_KEY=your_claude_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Development settings
EIDOLON_ENV=development
EIDOLON_LOG_LEVEL=DEBUG
EIDOLON_STORAGE_PATH=./dev_data

# Testing
EIDOLON_TEST_MODE=true
EIDOLON_MOCK_AI=true
```

## Installation

### Development Installation

```bash
# Install in editable mode with development dependencies
pip install -e ".[dev,test]"

# Alternative: Install from requirements files
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Verify Installation

```bash
# Check Python imports
python -c "import eidolon; print('✓ Eidolon imported successfully')"

# Check CLI access
python -m eidolon version

# Check core components
python -c "
from eidolon.core.observer import Observer
from eidolon.core.analyzer import Analyzer
from eidolon.core.memory import MemorySystem
print('✓ All core components available')
"

# Check OCR functionality
python -c "
from eidolon.core.analyzer import Analyzer
analyzer = Analyzer()
print(f'✓ Tesseract available: {analyzer._check_tesseract_available()}')
"
```

## Configuration

### Development Configuration

Create a development configuration file:

```bash
# Create development config directory
mkdir -p eidolon/config/

# Copy default configuration
cp eidolon/config/settings.yaml eidolon/config/settings-dev.yaml
```

Edit `settings-dev.yaml` for development:

```yaml
# Development-specific settings
observer:
  capture_interval: 5           # Faster for testing
  storage_path: "./dev_data"    # Separate dev data
  max_storage_gb: 1.0          # Smaller limit for dev
  
analysis:
  routing:
    local_first: true          # Prefer local models in dev
    cost_limit_daily: 1.0      # Lower cost limit
    
memory:
  db_path: "./dev_data/dev.db" # Separate dev database

logging:
  level: "DEBUG"               # Verbose logging
  file_path: "./logs/dev.log"

development:
  debug_mode: true
  mock_ai_responses: false     # Set to true to avoid API calls
  save_debug_screenshots: true
  verbose_logging: true
```

### IDE Setup

#### VS Code
Install recommended extensions:

```json
// .vscode/extensions.json
{
    "recommendations": [
        "ms-python.python",
        "ms-python.flake8",
        "ms-python.black-formatter",
        "ms-python.isort",
        "ms-toolsai.jupyter",
        "charliermarsh.ruff"
    ]
}
```

Configure VS Code settings:

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./eidolon_env/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "100"],
    "python.sortImports.args": ["--profile", "black"],
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true,
        "htmlcov": true
    }
}
```

#### PyCharm
1. Open project in PyCharm
2. Configure interpreter: Settings → Project → Python Interpreter
3. Point to `./eidolon_env/bin/python`
4. Enable pytest: Settings → Tools → Python Integrated Tools → Testing
5. Configure code style: Settings → Editor → Code Style → Python

## Development Workflow

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test
python -m pytest tests/ -v

# Run code quality checks
black eidolon/ tests/
flake8 eidolon/ tests/
python -m pytest --cov=eidolon tests/

# Commit changes
git add .
git commit -m "feat: add your feature description"

# Push and create PR
git push origin feature/your-feature-name
```

### 2. Testing Changes

```bash
# Run specific test file
python -m pytest tests/unit/test_observer.py -v

# Run with coverage
python -m pytest --cov=eidolon --cov-report=html tests/

# Run integration tests
python -m pytest tests/integration/ -v

# Run all tests
python -m pytest tests/ -v --tb=short
```

### 3. Code Quality

```bash
# Format code
black eidolon/ tests/

# Sort imports
isort eidolon/ tests/

# Lint code
flake8 eidolon/ tests/

# Type checking
mypy eidolon/

# Security check
bandit -r eidolon/
```

### 4. Running the Application

```bash
# Start monitoring in development mode
EIDOLON_ENV=development python -m eidolon capture --verbose

# Use development config
python -m eidolon --config eidolon/config/settings-dev.yaml capture

# Test specific functionality
python -m eidolon search "test query" --format json
```

## Testing

### Test Structure

```
tests/
├── unit/                  # Unit tests for individual components
│   ├── test_observer.py
│   ├── test_analyzer.py
│   ├── test_memory.py
│   └── test_storage.py
├── integration/          # Integration tests
│   ├── test_full_workflow.py
│   └── test_api_integration.py
├── fixtures/             # Test data and fixtures
│   ├── sample_screenshots/
│   └── mock_responses/
└── conftest.py          # Pytest configuration
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test category
python -m pytest tests/unit/
python -m pytest tests/integration/

# Run with coverage
python -m pytest --cov=eidolon --cov-report=html --cov-report=term

# Run tests in parallel
python -m pytest -n auto

# Run with verbose output
python -m pytest -v --tb=short

# Run specific test
python -m pytest tests/unit/test_observer.py::TestObserver::test_start_stop_monitoring
```

### Writing Tests

Example unit test:

```python
import pytest
from unittest.mock import Mock, patch
from eidolon.core.observer import Observer

class TestObserver:
    @pytest.fixture
    def mock_config(self):
        return {
            "capture_interval": 1,
            "storage_path": "/tmp/test"
        }
    
    @pytest.fixture
    def observer(self, mock_config):
        return Observer(config_override=mock_config)
    
    def test_observer_initialization(self, observer):
        assert observer is not None
        assert not observer._running
    
    @patch('eidolon.core.observer.mss.mss')
    def test_capture_screenshot(self, mock_mss, observer):
        # Setup mock
        mock_mss.return_value.__enter__.return_value.grab.return_value = Mock()
        
        # Test
        screenshot = observer.capture_screenshot()
        
        # Assert
        assert screenshot is not None
        assert screenshot.width > 0
        assert screenshot.height > 0
```

### Test Configuration

```python
# conftest.py
import pytest
import tempfile
from pathlib import Path

@pytest.fixture(scope="session")
def temp_data_dir():
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def mock_config(temp_data_dir):
    """Standard test configuration."""
    return {
        "storage_path": str(temp_data_dir),
        "db_path": str(temp_data_dir / "test.db"),
        "capture_interval": 1,
        "max_storage_gb": 0.1
    }
```

## Code Quality

### Pre-commit Hooks

Install pre-commit hooks:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

Pre-commit configuration (`.pre-commit-config.yaml`):

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.9.1
    hooks:
      - id: black
        language_version: python3

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        args: ["--max-line-length=100"]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.6.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

### Code Standards

Follow these coding standards:

1. **PEP 8**: Python style guide
2. **Black**: Code formatting (line length: 100)
3. **isort**: Import sorting
4. **Type hints**: Use throughout codebase
5. **Docstrings**: Google style documentation

Example code style:

```python
from typing import Dict, List, Optional, Union
from pathlib import Path

class ExampleClass:
    """Example class demonstrating code standards.
    
    This class shows proper formatting, type hints, and documentation
    following project standards.
    
    Args:
        config: Configuration dictionary
        data_path: Path to data directory
    """
    
    def __init__(self, config: Dict[str, Any], data_path: Path) -> None:
        self.config = config
        self.data_path = data_path
        self._cache: Dict[str, Any] = {}
    
    def process_data(
        self, 
        input_data: List[str], 
        options: Optional[Dict[str, Any]] = None
    ) -> Union[str, None]:
        """Process input data with optional configuration.
        
        Args:
            input_data: List of input strings to process
            options: Optional processing configuration
            
        Returns:
            Processed result string, or None if processing failed
            
        Raises:
            ValueError: If input_data is empty
        """
        if not input_data:
            raise ValueError("Input data cannot be empty")
        
        # Implementation here
        return "processed_result"
```

## Debugging

### Logging Configuration

Enable detailed logging for debugging:

```python
import logging

# Set debug level
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Component-specific logging
logger = logging.getLogger('eidolon.observer')
logger.setLevel(logging.DEBUG)
```

### Debug Tools

#### 1. Interactive Debugging

```python
# Insert breakpoint in code
import pdb; pdb.set_trace()

# Or use built-in breakpoint (Python 3.7+)
breakpoint()
```

#### 2. Performance Profiling

```bash
# Profile script execution
python -m cProfile -o profile.stats -m eidolon capture

# Analyze profile
python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
"
```

#### 3. Memory Monitoring

```python
import tracemalloc

# Start memory tracing
tracemalloc.start()

# Your code here
observer = Observer()
observer.start_monitoring()

# Get memory usage
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
```

### Common Debug Scenarios

#### 1. Screenshot Capture Issues

```python
# Debug screenshot capture
from eidolon.core.observer import Observer

observer = Observer()

# Test basic capture
try:
    screenshot = observer.capture_screenshot()
    print(f"✓ Capture successful: {screenshot.width}x{screenshot.height}")
except Exception as e:
    print(f"✗ Capture failed: {e}")

# Test monitor detection
try:
    monitors = observer._get_monitor_info()
    print(f"✓ Detected {len(monitors['all_monitors'])} monitors")
except Exception as e:
    print(f"✗ Monitor detection failed: {e}")
```

#### 2. OCR/Analysis Issues

```python
# Debug text extraction
from eidolon.core.analyzer import Analyzer

analyzer = Analyzer()

# Test Tesseract availability
if analyzer._check_tesseract_available():
    print("✓ Tesseract available")
else:
    print("✗ Tesseract not available")

# Test text extraction
try:
    result = analyzer.extract_text("test_image.png")
    print(f"✓ Extracted: {len(result.text)} characters")
except Exception as e:
    print(f"✗ Text extraction failed: {e}")
```

#### 3. Database Issues

```python
# Debug database operations
from eidolon.storage.metadata_db import MetadataDatabase

try:
    db = MetadataDatabase()
    stats = db.get_statistics()
    print(f"✓ Database connection successful: {stats}")
except Exception as e:
    print(f"✗ Database error: {e}")
```

## Contributing

### Contribution Workflow

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Make changes** following code standards
4. **Add tests** for new functionality
5. **Update documentation** if needed
6. **Run test suite**: `python -m pytest`
7. **Commit changes**: `git commit -m 'feat: add amazing feature'`
8. **Push to branch**: `git push origin feature/amazing-feature`
9. **Create Pull Request**

### Commit Message Format

Use conventional commit format:

```
type(scope): short description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Maintenance tasks

Examples:
```
feat(observer): add support for multi-monitor capture
fix(analyzer): handle corrupted image files gracefully
docs(api): update API documentation for memory system
test(storage): add integration tests for vector database
```

### Code Review Guidelines

- Ensure all tests pass
- Maintain/improve test coverage
- Follow code style guidelines
- Update documentation for API changes
- Add type hints for all functions
- Include proper error handling

## Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# Issue: ModuleNotFoundError
# Solution: Verify installation
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

#### 2. Tesseract Not Found

```bash
# macOS
brew install tesseract

# Ubuntu
sudo apt install tesseract-ocr

# Windows: Download from GitHub releases
# Ensure tesseract is in PATH
tesseract --version
```

#### 3. Permission Errors (macOS)

```bash
# Grant screen recording permissions
# System Preferences → Security & Privacy → Privacy → Screen Recording
# Add Terminal/IDE to allowed applications
```

#### 4. Database Lock Errors

```bash
# Check for orphaned processes
ps aux | grep eidolon

# Remove lock files
rm ./data/*.lock

# Reset database
rm ./data/eidolon.db
```

#### 5. Memory Issues

```bash
# Monitor memory usage
top -p $(pgrep -f eidolon)

# Reduce memory usage
# Set smaller batch sizes in config
# Enable local_only_mode to avoid loading cloud models
```

### Getting Help

1. **Check the logs**: `tail -f logs/eidolon.log`
2. **Run diagnostics**: `python -m eidolon status --verbose`
3. **Check configuration**: `python -c "from eidolon.utils.config import get_config; print(get_config())"`
4. **Verify dependencies**: `pip check`
5. **Create GitHub issue** with:
   - Error message and stack trace
   - System information
   - Configuration (sanitized)
   - Steps to reproduce

### Performance Optimization

#### Development Environment

```yaml
# Optimized dev config
observer:
  capture_interval: 10       # Longer intervals
  max_cpu_percent: 20        # Higher limit for dev
  max_memory_mb: 2048       # Higher limit for dev

analysis:
  local_models:
    enable_caching: true     # Cache models in memory
  routing:
    local_first: true        # Prefer local models

memory:
  batch_size: 100           # Larger batches for better performance
```

#### Resource Monitoring

```python
# Monitor resource usage during development
import psutil
import time

def monitor_resources():
    process = psutil.Process()
    while True:
        cpu = process.cpu_percent()
        memory = process.memory_info().rss / 1024 / 1024
        print(f"CPU: {cpu:.1f}%, Memory: {memory:.1f}MB")
        time.sleep(5)

# Run in separate thread
import threading
threading.Thread(target=monitor_resources, daemon=True).start()
```

This developer setup guide provides comprehensive instructions for setting up and contributing to the Eidolon project. Follow these guidelines to ensure a smooth development experience.