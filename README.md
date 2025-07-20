# Eidolon AI Personal Assistant

A hyper-personalized AI assistant that functions as a digital twin by monitoring computer activity, building comprehensive knowledge bases, and acting autonomously on your behalf.

## 🚀 Features

- **Silent Monitoring**: Continuous screenshot capture with intelligent change detection
- **AI-Powered Analysis**: Florence-2 vision model and local AI for advanced content understanding
- **Intelligent OCR**: Tesseract and EasyOCR for accurate text extraction
- **Scene Classification**: Automatic categorization (development, web browsing, documents, etc.)
- **Smart Search**: Full-text search with SQLite FTS5 and semantic capabilities
- **Vision Analysis**: Object detection, UI element recognition, and scene understanding
- **Privacy First**: Local-first architecture with user control over data
- **Cross-Platform**: Support for Windows, macOS, and Linux

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- Tesseract OCR (for text extraction)
- **At least 8GB RAM** (for AI models - 16GB recommended for optimal performance)
- **At least 20GB disk space** (AI models require additional storage)
- Internet connection for initial AI model downloads

### Quick Install

```bash
# Clone the repository
git clone https://github.com/eidolon-ai/eidolon.git
cd eidolon

# Create virtual environment (recommended)
python3 -m venv eidolon_env
source eidolon_env/bin/activate  # On Windows: eidolon_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Development Setup

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

For detailed installation instructions, see [Installation Guide](docs/user-guide/INSTALL.md).

### Tesseract Installation

#### macOS
```bash
brew install tesseract
```

#### Ubuntu/Debian
```bash
sudo apt-get install tesseract-ocr
```

#### Windows
Download from: https://github.com/UB-Mannheim/tesseract/wiki

## 🏃‍♂️ Quick Start

### 1. Start Screenshot Monitoring

```bash
# Activate virtual environment first
source eidolon_env/bin/activate

# Start with default settings
python -m eidolon.cli.main capture

# Custom interval (every 30 seconds)
python -m eidolon.cli.main capture --interval 30

# Run in background
python -m eidolon.cli.main capture --background
```

### 2. Search Your Activity

```bash
# Search for content
python -m eidolon.cli.main search "python code"

# Limit results
python -m eidolon.cli.main search "meetings" --limit 5

# JSON output
python -m eidolon.cli.main search "documents" --format json

# Filter by content type
python -m eidolon.cli.main search "text" --content-type document
```

### 3. Check System Status

```bash
# View current status
python -m eidolon.cli.main status

# JSON format for scripting
python -m eidolon.cli.main status --json
```

### 4. Data Management

```bash
# Clean up old data (keep 30 days)
python -m eidolon.cli.main cleanup --days 30

# Export your data
python -m eidolon.cli.main export --path my_data.json

# View version info
python -m eidolon.cli.main version
```

## 💡 Example Workflow

Here's a complete example of using Eidolon:

```bash
# 1. Set up the environment
source eidolon_env/bin/activate

# 2. Start monitoring your screen (in terminal 1)
python -m eidolon.cli.main capture --interval 5

# 3. Do some work on your computer...
# - Open some documents
# - Browse websites  
# - Write code
# - Read emails

# 4. After a few minutes, stop monitoring (Ctrl+C) and search (in terminal 2)
python -m eidolon.cli.main search "python"
# Returns: Found 3 results with Python-related content

python -m eidolon.cli.main search "email" --format json
# Returns: JSON with detailed metadata about email-related screenshots

python -m eidolon.cli.main status
# Shows: "29 screenshots captured, 25 with text extracted"

# 5. View database statistics
python -c "
from eidolon.storage.metadata_db import MetadataDatabase
db = MetadataDatabase()
stats = db.get_statistics()
print(f'📊 Database Stats:')
print(f'   Screenshots: {stats[\"total_screenshots\"]}')
print(f'   With OCR text: {stats[\"screenshots_with_text\"]}') 
print(f'   Content types: {list(stats[\"content_types\"].keys())}')
"
```

### Sample Output

```
📊 Database Stats:
   Screenshots: 29
   With OCR text: 25
   Content types: ['document', 'terminal', 'browser']

🔍 Search Results for "python":
2025-07-19 15:06:27 | document     | 0.88       | Terminal output showing Python script execution
2025-07-19 15:04:12 | browser      | 0.92       | Python documentation page on functions  
2025-07-19 15:02:03 | document     | 0.95       | Code editor with Python file open

🤖 AI Analysis Results:
Scene Type: development
Vision Model: florence-2  
Description: "The image shows a computer screen with code editor interface displaying Python syntax..."
Objects Detected: window, text_editor, menu_bar
UI Elements: 3 detected (buttons, menus, text fields)
```

## ⚙️ Configuration

Eidolon uses a YAML configuration file located at `config/settings.yaml`. Key settings include:

```yaml
observer:
  capture_interval: 10          # Seconds between screenshots
  activity_threshold: 0.05      # Change detection sensitivity
  max_storage_gb: 50           # Storage limit

privacy:
  local_only_mode: false       # Never send to cloud APIs
  auto_redaction: true         # Auto-hide sensitive info
  data_retention_days: 365     # How long to keep data

analysis:
  local_models:
    vision: "microsoft/florence-2-base"
    clip: "openai/clip-vit-base-patch32"
  cloud_apis:
    gemini_key: "${GEMINI_API_KEY}"
    claude_key: "${CLAUDE_API_KEY}"
    openrouter_key: "${OPENROUTER_API_KEY}"  # OpenRouter.ai integration
```

### Environment Variables

Set API keys for cloud AI features:

```bash
export GEMINI_API_KEY="your_gemini_key"
export CLAUDE_API_KEY="your_claude_key"
export OPENROUTER_API_KEY="your_openrouter_key"  # Cost-effective Claude access
export OPENAI_API_KEY="your_openai_key"
```

💡 **OpenRouter.ai Support**: Native integration with [OpenRouter.ai](https://openrouter.ai/) for cost-effective access to Claude and other AI models.

## 📁 Project Structure

```
eidolon/
├── eidolon/               # Main source code
│   ├── cli/               # Command-line interface
│   ├── core/              # Core components (observer, analyzer, memory)
│   ├── models/            # AI model integrations
│   ├── storage/           # Data storage and management
│   ├── utils/             # Shared utilities
│   └── config/            # Configuration files
│       ├── settings.yaml  # Main configuration
│       └── logging.yaml   # Logging configuration
├── tests/                 # Organized test suite
│   ├── phase/             # Phase-based integration tests
│   │   ├── test_phase1.py # Foundation and observer tests
│   │   ├── test_phase2.py # Analysis system tests
│   │   ├── test_phase3.py # Local AI integration tests
│   │   ├── test_phase4.py # Cloud AI and vector DB tests
│   │   └── test_phase5.py # Advanced analytics tests
│   ├── unit/              # Unit tests for components
│   ├── integration/       # System integration tests
│   └── fixtures/          # Test data and utilities
├── docs/                  # Comprehensive documentation
│   ├── user-guide/        # User documentation
│   ├── development/       # Development guides
│   ├── api/               # API documentation
│   └── examples/          # Usage examples
├── data/                  # Generated data (gitignored)
├── logs/                  # Log files (gitignored)
├── pyproject.toml         # Project configuration
└── requirements.txt       # Dependencies
```

## 🏗️ Architecture

Eidolon consists of four main components:

### Observer
- **Screenshot Capture**: Intelligent capture with change detection
- **Activity Monitoring**: Keyboard, mouse, and window tracking
- **Resource Management**: CPU and memory usage optimization

### Analyzer
- **OCR Text Extraction**: Tesseract and EasyOCR integration
- **Local AI Models**: Florence-2, CLIP for image understanding
- **Cloud AI APIs**: Gemini, Claude, GPT-4V for deep analysis

### Memory
- **Vector Database**: ChromaDB for semantic search
- **Metadata Storage**: SQLite for structured data
- **Knowledge Base**: Relationship mapping and context preservation

### Interface
- **CLI Commands**: Full command-line interface
- **Natural Language**: Query your data conversationally
- **Web Interface**: Optional browser-based UI
- **API Access**: REST API for integrations

## 🔒 Privacy & Security

Eidolon is designed with privacy as a core principle:

- **Local First**: All processing happens locally by default
- **User Control**: Complete control over what data is sent to cloud APIs
- **Auto-Redaction**: Automatically hides sensitive information
- **Encryption**: All data encrypted at rest
- **No Tracking**: No telemetry or data collection
- **Open Source**: Full transparency in what the code does

### Sensitive Data Handling

- Passwords, API keys, and secrets automatically redacted
- Password managers and secure apps excluded from monitoring
- Configurable patterns for additional sensitive content
- Option to run in completely offline mode

## 📊 Current Status

**Phase 1: Foundation** ✅ *Complete* (100% tested)
- Project setup and configuration system
- Basic screenshot capture with change detection
- CLI interface and testing framework
- Resource monitoring and performance optimization

**Phase 2: Intelligent Capture** ✅ *Complete* (100% tested)
- Smart change detection algorithms with advanced metrics
- OCR text extraction (Tesseract + EasyOCR)
- Content categorization and analysis
- SQLite database with FTS5 full-text search
- Enhanced storage and search capabilities

**Phase 3: Local AI Integration** ✅ *Complete* (100% tested)
- Florence-2 vision model for advanced image understanding
- AI-enhanced content analysis with intelligent fallbacks
- Advanced scene classification (development, web browsing, etc.)
- Object and UI element detection capabilities
- Vision data storage and retrieval

**Phase 4: Cloud AI & Semantic Memory** ✅ *Complete* (100% tested)
- ChromaDB vector database with 384-dimensional embeddings
- Cloud AI API integration (Gemini, Claude, OpenAI)
- Natural language query processing with intent recognition
- RAG (Retrieval-Augmented Generation) implementation
- Semantic and hybrid search capabilities
- Intelligent local/cloud routing with cost optimization

**Phase 5: Advanced Analytics** ⏳ *Planned*
- Productivity insights and time tracking
- Personal pattern recognition
- Project timeline reconstruction
- Advanced reporting

**Phase 6: MCP Integration & Agency** ⏳ *Planned*
- Model Context Protocol server
- Basic autonomous actions
- Tool orchestration
- Safety mechanisms

**Phase 7: Digital Twin** ⏳ *Planned*
- Advanced agentic capabilities
- Proactive assistance
- Style replication
- Complete system integration

## 🧪 Testing

### Phase Validation Tests

```bash
# Test Phase 1 (Foundation) - All 7 tests
source eidolon_env/bin/activate
python test_phase1.py

# Test Phase 2 (Intelligent Capture) - All 7 tests  
python test_phase2.py

# Test Phase 3 (Local AI Integration) - All 7 tests
python test_phase3.py

# Run individual component tests
python -c "from eidolon.core.observer import Observer; print('Observer working!')"
python -c "from eidolon.core.analyzer import Analyzer; print('Analyzer working!')"
python -c "from eidolon.storage.metadata_db import MetadataDatabase; print('Database working!')"
```

### AI Model Testing

```bash
# Test Florence-2 vision model loading
python -c "
from eidolon.core.analyzer import Analyzer
analyzer = Analyzer()
print(f'Florence-2 available: {analyzer._florence_available}')
print(f'Model loaded: {analyzer._florence_model is not None}')
"

# Test AI-enhanced content analysis
python -c "
from eidolon.core.analyzer import Analyzer
from pathlib import Path
analyzer = Analyzer()
screenshots = list(Path('data/screenshots').glob('*.png'))
if screenshots:
    analysis = analyzer.analyze_content(screenshots[0])
    print(f'Content type: {analysis.content_type}')
    if analysis.vision_analysis:
        print(f'AI model used: {analysis.vision_analysis.model_used}')
        print(f'Scene type: {analysis.vision_analysis.scene_type}')
"

# Test scene classification
python -c "
from eidolon.core.analyzer import Analyzer
analyzer = Analyzer()
scene = analyzer._classify_scene_type('code editor with terminal', [])
print(f'Scene classification: {scene}')
"
```

### Integration Testing

```bash
# Test complete workflow (Phase 1 + Phase 2)
python -c "
from eidolon.core.observer import Observer
from eidolon.storage.metadata_db import MetadataDatabase
import time
observer = Observer({'capture_interval': 1})
observer.start_monitoring()
time.sleep(3)
observer.stop_monitoring()
db = MetadataDatabase()
stats = db.get_statistics()
print(f'Screenshots: {stats[\"total_screenshots\"]}, With text: {stats[\"screenshots_with_text\"]}')
"

# Test OCR functionality
python -c "
from eidolon.core.analyzer import Analyzer
from pathlib import Path
analyzer = Analyzer()
screenshots = list(Path('data/screenshots').glob('*.png'))
if screenshots:
    result = analyzer.extract_text(screenshots[0])
    print(f'OCR: {result.word_count} words, {result.confidence:.2f} confidence')
"

# Test database search
python -c "
from eidolon.storage.metadata_db import MetadataDatabase
db = MetadataDatabase()
results = db.search_text('terminal')
print(f'Search results: {len(results)} matches')
"
```

### Development Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov

# Run specific test file
pytest tests/test_observer.py

# Run with verbose output
pytest -v
```

### CLI Testing Commands

```bash
# Test all CLI commands
python -m eidolon.cli.main --help
python -m eidolon.cli.main version
python -m eidolon.cli.main status
python -m eidolon.cli.main search "test" --limit 3
python -m eidolon.cli.main search "terminal" --format json
python -m eidolon.cli.main search "document" --content-type document
```

## 🔧 Troubleshooting

### Common Issues

#### System Requirements Check
```bash
# Check system specifications for AI models
python -c "
import psutil
import shutil

print('🖥️  SYSTEM REQUIREMENTS CHECK')
print('=' * 40)

# Check RAM
total_memory = psutil.virtual_memory().total / (1024**3)
available_memory = psutil.virtual_memory().available / (1024**3)
print(f'Total RAM: {total_memory:.1f}GB')
print(f'Available RAM: {available_memory:.1f}GB')

if total_memory >= 16:
    print('✅ Excellent - 16GB+ RAM (optimal for AI)')
elif total_memory >= 8:
    print('✅ Good - 8GB+ RAM (sufficient for AI)')
else:
    print('⚠️  Warning - <8GB RAM (may limit AI performance)')

# Check disk space
total, used, free = shutil.disk_usage('.')
free_gb = free / (1024**3)
print(f'Free disk space: {free_gb:.1f}GB')

if free_gb >= 20:
    print('✅ Sufficient disk space')
else:
    print('⚠️  Warning - <20GB free space (may limit AI models)')

# Check CPU cores
cpu_count = psutil.cpu_count()
print(f'CPU cores: {cpu_count}')
print('✅ System check complete')
"
```

#### Virtual Environment Issues
```bash
# If virtual environment is not activated
source eidolon_env/bin/activate

# If environment doesn't exist
python3 -m venv eidolon_env
source eidolon_env/bin/activate
pip install -r requirements.txt
```

#### Tesseract OCR Issues
```bash
# Check if Tesseract is installed
tesseract --version

# macOS installation
brew install tesseract

# Ubuntu/Debian installation  
sudo apt-get install tesseract-ocr

# Test OCR functionality
python -c "from eidolon.core.analyzer import Analyzer; a = Analyzer(); print(f'Tesseract available: {a._tesseract_available}')"
```

#### AI Model Issues
```bash
# Install missing AI dependencies
pip install einops timm accelerate

# Test Florence-2 model loading (will download model on first use)
python -c "from eidolon.core.analyzer import Analyzer; a = Analyzer(); print(f'Florence-2: {a._florence_available}')"

# If model download fails, check internet connection and disk space
# Florence-2 model is ~230MB and downloads to ~/.cache/huggingface/
# Total AI models cache: ~500MB

# Check available disk space
df -h ~/.cache/huggingface/

# Force model re-download if corrupted
rm -rf ~/.cache/huggingface/transformers/models--microsoft--Florence-2-base

# Check system memory for AI models
python -c "
import psutil
total_memory = psutil.virtual_memory().total / (1024**3)
print(f'Total system memory: {total_memory:.1f}GB')
if total_memory >= 8:
    print('✅ Sufficient memory for AI models')
else:
    print('⚠️  Recommended: 8GB+ RAM for optimal AI performance')
"
```

#### Permission Issues (macOS)
- Go to System Preferences → Security & Privacy → Privacy → Screen Recording
- Add Terminal (or your terminal application) to the list
- Restart terminal and try again

#### Module Import Errors
```bash
# Ensure you're in the right directory and virtual environment
cd /path/to/eidolon
source eidolon_env/bin/activate
python -c "import sys; print(sys.path)"
```

#### Database Issues
```bash
# Check database status
python -c "
from eidolon.storage.metadata_db import MetadataDatabase
db = MetadataDatabase()
stats = db.get_statistics()
print('Database working:', stats)
"

# Reset database if needed (WARNING: deletes all data)
rm -rf data/memory/metadata.db
```

### Validation Commands
```bash
# Quick health check for all phases (including AI)
python -c "
try:
    from eidolon.core.observer import Observer
    from eidolon.core.analyzer import Analyzer  
    from eidolon.storage.metadata_db import MetadataDatabase
    print('✅ All core modules importable')
    
    # Test Observer
    observer = Observer()
    print('✅ Observer initialized')
    
    # Test Analyzer with AI capabilities
    analyzer = Analyzer()
    print(f'✅ Analyzer initialized')
    print(f'   - Tesseract: {analyzer._tesseract_available}')
    print(f'   - Florence-2: {analyzer._florence_available}')
    
    # Test Database
    db = MetadataDatabase()
    stats = db.get_statistics()
    print(f'✅ Database working ({stats[\"total_screenshots\"]} screenshots)')
    
    print('🎉 All systems operational with AI capabilities!')
except Exception as e:
    print(f'❌ Error: {e}')
"
```

## 📈 Performance

### Resource Usage (with AI Models)
- **Memory**: 1.5GB baseline with Florence-2 loaded, up to 8GB during intensive analysis
- **CPU**: <5% idle, up to 20% during AI analysis
- **Storage**: 1-2GB per week typical usage (includes model cache)
- **Network**: <100MB per day for cloud analysis, ~500MB for initial model downloads

### Performance Modes
```yaml
# Standard Mode (config/settings.yaml)
observer:
  max_memory_mb: 8192        # 8GB for AI models
  max_cpu_percent: 20.0      # Higher for AI processing

# High-Performance Mode (for 16GB+ systems)
observer:
  max_memory_mb: 16384       # 16GB for multiple models
  max_cpu_percent: 30.0      # More aggressive processing
```

### Performance Monitoring
```bash
# Check current resource usage
python -m eidolon.cli.main status

# Monitor during capture session
python -c "
from eidolon.core.observer import Observer
import time
observer = Observer()
observer.start_monitoring()
time.sleep(10)
status = observer.get_status()
print(f'Memory: {status[\"performance_metrics\"][\"memory_usage_mb\"]:.1f}MB')
print(f'CPU: {status[\"performance_metrics\"][\"cpu_usage_percent\"]:.1f}%')
observer.stop_monitoring()
"
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/development/CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to your branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use Black for code formatting
- Add type hints to all functions
- Write docstrings for public APIs
- Maintain >90% test coverage

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [https://docs.eidolon.ai](https://docs.eidolon.ai)
- **Issues**: [GitHub Issues](https://github.com/eidolon-ai/eidolon/issues)
- **Discussions**: [GitHub Discussions](https://github.com/eidolon-ai/eidolon/discussions)
- **Security**: Email security@eidolon.ai for security issues

## 🙏 Acknowledgments

- Built with [Claude Code](https://claude.ai/code) for AI-assisted development
- Uses amazing open-source projects like Transformers, ChromaDB, and FastAPI
- Inspired by the vision of truly personal AI assistants

---

**⚠️ Note**: Eidolon is currently in active development. Features and APIs may change rapidly. See our [Progress Plan](docs/development/PROGRESS_PLAN.md) for detailed development status.

## 📚 Documentation

- **[User Guide](docs/user-guide/INSTALL.md)**: Installation and setup instructions
- **[Development Guide](docs/development/PROGRESS_PLAN.md)**: Development roadmap and status
- **[API Documentation](docs/api/)**: Coming in Phase 4
- **[Examples](docs/examples/)**: Usage examples and tutorials