# Eidolon AI Personal Assistant

🤖 A sophisticated AI personal assistant that continuously monitors your screen, performs intelligent analysis, and provides semantic search capabilities through your digital activities.

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.9+-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## ✨ Features

- 📸 **Continuous Screen Monitoring**: Intelligent screenshot capture with change detection
- 🧠 **AI-Powered Analysis**: Advanced vision models (Florence-2) for content understanding
- 📝 **OCR & Text Extraction**: Multi-engine OCR with Tesseract and EasyOCR
- 🔍 **Semantic Search**: Vector-based search through your digital history with ChromaDB
- 🔒 **Privacy-First**: Local processing with optional cloud AI integration
- ⚡ **Real-Time Processing**: Concurrent processing pipeline with health monitoring
- 💬 **Interactive Chat**: Natural language queries about your screen activity
- 🌐 **Cloud AI Support**: Optional integration with Gemini, Claude, and OpenAI

## 🚀 Quick Start

### Option 1: Automated Installation (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd eidolon

# Run the installation script
./scripts/install.sh

# Start the system
./scripts/start.sh
```

### Option 2: Manual Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (optional)
   ```

3. **Start the System**

   **For 8GB RAM systems:**
   ```bash
   python -m eidolon start --low-memory --interval 60
   ```
   
   **For 16GB RAM systems:**
   ```bash
   python -m eidolon start --memory-limit 12.0
   ```
   
   **For 32GB+ RAM systems:**
   ```bash
   python -m eidolon start
   ```

## 📚 Usage

### Basic Commands

```bash
# Start the complete system (auto-detects RAM)
python -m eidolon start

# Start with memory optimization
python -m eidolon start --low-memory --background

# Search your digital history
python -m eidolon search "meeting notes"
python -m eidolon search "Python code from yesterday"

# Interactive chat with your data
python -m eidolon chat

# Check system status and memory usage
python -m eidolon status

# Stop the system
python -m eidolon stop

# Clean up old data to free memory
python -m eidolon cleanup --days 30
```

### Memory Optimization

**If you experience high memory usage or crashes:**

```bash
# Emergency low-memory mode
export EIDOLON_USE_CPU_ONLY=1
python -m eidolon start --low-memory --interval 120

# Disable local vision processing
export EIDOLON_DISABLE_LOCAL_VISION=1
python -m eidolon start --memory-limit 4.0
```

**For detailed memory optimization:** See [Memory Optimization Guide](docs/memory_optimization.md)

### Scripts and Utilities

```bash
# Complete installation
./scripts/install.sh

# Start all services
./scripts/start.sh

# System health check
./scripts/health_check.sh
```

### Example Queries

```bash
# Search for specific content
python -m eidolon search "error messages from today"
python -m eidolon search "emails about the project"
python -m eidolon search "terminal commands I ran"

# Interactive chat examples
# "What was I working on this morning?"
# "Show me any Python code from the last hour"
# "Find all the websites I visited yesterday"
```

## ⚙️ Configuration

The system uses environment variables for configuration. Copy `.env.example` to `.env` and customize:

```bash
# Cloud AI API Keys (optional - for enhanced features)
GEMINI_API_KEY=your_gemini_key_here
ANTHROPIC_API_KEY=your_claude_key_here
OPENAI_API_KEY=your_openai_key_here

# System Configuration
EIDOLON_DATA_DIR=./data
LOG_LEVEL=INFO
CAPTURE_INTERVAL=30
ACTIVITY_THRESHOLD=0.1
MAX_STORAGE_GB=10

# Privacy Settings
DATA_RETENTION_DAYS=90
AUTO_REDACT_SENSITIVE=true
```

## 🏗️ Architecture

Eidolon follows a clean, modular architecture:

```
eidolon/
├── core/           # Core processing components
│   ├── observer.py     # Screenshot capture and monitoring
│   ├── analyzer.py     # AI-powered content analysis
│   ├── memory.py       # Semantic memory system
│   └── interface.py    # User interaction layer
├── storage/        # Data persistence
│   ├── vector_db.py    # ChromaDB vector storage
│   └── metadata_db.py  # SQLite metadata storage
├── utils/          # Utilities and configuration
└── models/         # AI model integrations
```

### Key Components

- **Observer**: Continuous screenshot capture with intelligent change detection
- **Analyzer**: AI-powered analysis using Florence-2 vision model and OCR
- **Memory System**: ChromaDB vector database with semantic search capabilities
- **Interface**: Natural language query processing and chat functionality
- **Storage**: Dual storage system (vector + metadata) for optimal performance

## 📋 Requirements

- **Python**: 3.9+ (3.11+ recommended)
- **OS**: macOS/Linux (Windows support planned)
- **Memory**: 4GB+ RAM recommended
- **Storage**: 2GB+ free space for data
- **GPU**: Optional (improves AI processing speed)
- **Permissions**: Screen recording permissions (macOS)

## 🔒 Privacy & Security

- **Local-First**: All processing happens locally by default
- **Optional Cloud**: Cloud APIs are optional and user-controlled
- **Data Encryption**: Screenshots and sensitive data are encrypted
- **Auto-Redaction**: Sensitive information is automatically detected and redacted
- **Retention Control**: Configurable data retention policies
- **No Telemetry**: No data is sent to external services without explicit consent

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_core.py -v
python -m pytest tests/test_cli.py -v

# Check system health
./scripts/health_check.sh
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🆘 Support

If you encounter issues:

1. Check the [health check script](scripts/health_check.sh)
2. Review the logs in `data/logs/`
3. Run tests to identify issues
4. Check system requirements and permissions

---

**Made with ❤️ for digital productivity and AI-powered assistance**
