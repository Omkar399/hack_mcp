# 🖥️ Screen Memory Assistant

> **Local screen capture and memory with ML-powered search**

A privacy-preserving, fully local screen memory system that captures, processes, and searches everything you see on your screen using OCR, CLIP embeddings, and LLM fallbacks.

## 🌟 Features

- **📸 Continuous Screen Capture**: Automatic screenshot capture every 2 seconds
- **🔍 OCR Text Extraction**: Multiple OCR engines (Tesseract, EasyOCR) with fallbacks
- **🧠 Semantic Search**: CLIP-powered visual similarity search
- **🤖 Vision API Fallback**: GPT-4o Vision via Martian router for complex scenes
- **🗄️ Full-Text Search**: PostgreSQL with vector similarity (pgvector)
- **⚡ Real-time API**: FastAPI server with MCP-compatible endpoints
- **🖱️ CLI Interface**: Beautiful CLI tool with Rich output
- **⚡ Fast Setup**: uv-powered dependency management (10x faster than pip)
- **🐳 Containerized**: One-command Docker setup
- **🔒 Privacy-First**: Everything runs locally, no cloud data storage

## 🏗️ Architecture

```mermaid
flowchart TD
    A[PyAutoGUI Screenshot] --> B[Multi-OCR Pipeline]
    B --> C[CLIP Embeddings]
    C --> D[PostgreSQL + pgvector]
    D --> E[FastAPI Server]
    E --> F[CLI Tool]
    
    B --> G[GPT-4o Vision Fallback]
    G --> D
    
    H[Martian Router] --> G
```

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) - Fast Python package installer
- macOS or Linux (Windows support TBD)

#### Install uv (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1. Launch the System

```bash
# Clone and setup (if not already done)
git clone <repository>
cd screen-memory-assistant

# Start everything with one command
./start.sh

# Or with health check
./start.sh --test
```

### 2. Basic Usage

```bash
# Check system health
uv run python cli.py health
# Or use the installed script: uv run screen-memory health

# Take a screenshot now
uv run python cli.py capture

# Search your screen history
uv run python cli.py search "docker command"

# Show recent events
uv run python cli.py recent --hours 24

# Semantic search (requires CLIP)
uv run python cli.py semantic "terminal window"

# Get recent errors
uv run python cli.py errors --minutes 30
```

### 3. Advanced Usage

```bash
# Search specific app
uv run python cli.py search "error" --app "Chrome"

# Search recent timeframe
uv run python cli.py search "password" --since 60  # last 60 minutes

# Get system statistics
uv run python cli.py stats

# Find last Docker command
uv run python cli.py docker
```

## 📋 API Endpoints (MCP-Compatible)

The system exposes FastAPI endpoints compatible with Model Context Protocol:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/capture_now` | POST | Take immediate screenshot |
| `/find` | POST | Full-text search |
| `/search_semantic` | POST | Vector similarity search |
| `/recent_errors` | GET | Find recent error dialogs |
| `/last_docker` | GET | Get last Docker command |
| `/calendar_between` | GET | Calendar entries in timeframe |
| `/health` | GET | System health check |

### Example API Usage

```python
import httpx

async with httpx.AsyncClient() as client:
    # Capture screenshot
    response = await client.post("http://localhost:5003/capture_now")
    
    # Search events
    response = await client.post("http://localhost:5003/find", json={
        "query": "docker run",
        "limit": 10
    })
```

## 🧪 Testing

Run the comprehensive integration test suite:

```bash
uv run python test_integration.py
```

This tests:
- ✅ Database connectivity
- ✅ Screen capture pipeline
- ✅ OCR engines
- ✅ CLIP embeddings
- ✅ API endpoints
- ✅ Search functionality
- ✅ Performance benchmarks

## 🔧 Configuration

### Environment Variables

```bash
# Database connection
DATABASE_URL=postgresql+asyncpg://hack:hack123@localhost:5432/screenmemory

# Martian router for vision API
MARTIAN_URL=http://localhost:5333
OPENAI_API_BASE=http://localhost:5333/v1
OPENAI_API_KEY=localrouter

# CLI API endpoint
SCREEN_MEMORY_API=http://localhost:5003
```

### Docker Services

- **PostgreSQL**: Database with pgvector extension
- **Martian Router**: LLM routing for vision fallback
- **App Container**: Main Python application

## 📊 Performance

Typical performance on modern MacBook:

- **Screenshot + OCR**: ~150ms
- **CLIP Embedding**: ~25ms (with Metal)
- **Database Insert**: ~5ms
- **Text Search**: <10ms
- **Vector Search**: ~50ms

Memory usage: ~500MB base + ~200MB per CLIP model

## 🛠️ Development

### Manual Setup (without Docker)

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies with uv (much faster than pip!)
uv sync

# Optional: Install with ML support (CLIP embeddings)
uv sync --extra ml

# Optional: Install with better OCR support
uv sync --extra ocr

# Or install everything for development
uv sync --extra dev --extra ml --extra ocr

# Start PostgreSQL with pgvector
brew install postgresql@16 pgvector
brew services start postgresql@16
createdb screenmemory

# Initialize database
uv run python -c "
import asyncio
from database import db
asyncio.run(db.initialize())
"

# Start Martian router
docker run -p 5333:5333 ghcr.io/withmartian/router:latest

# Start API server
uv run uvicorn screen_api:app --reload --port 5003

# Test capture
uv run python -c "
import asyncio
from capture import ScreenCapture
capture = ScreenCapture()
print(asyncio.run(capture.capture_screen(save_image=False)))
"
```

### Project Structure

```
screen-memory-assistant/
├── pyproject.toml         # uv project configuration & dependencies
├── uv.lock                # Locked dependency versions
├── .python-version        # Python version specification
├── docker-compose.yml      # Full stack orchestration
├── Dockerfile             # Python app container
├── database/
│   └── init.sql           # Database schema
├── models.py              # SQLAlchemy models
├── capture.py             # Screen capture + ML pipeline
├── database.py            # Database operations
├── screen_api.py          # FastAPI server
├── cli.py                 # Command-line interface
├── test_integration.py    # Integration tests
├── quick_test.py          # Quick verification tests
├── start.sh               # Startup script
└── README.md              # This file
```

## 🚨 Troubleshooting

### Common Issues

**OCR not working**
```bash
# Check OCR engines
uv run python -c "from capture import ScreenCapture; print(ScreenCapture().health_check())"

# Install missing engines (they should already be in pyproject.toml)
uv sync
```

**Database connection failed**
```bash
# Check PostgreSQL status
docker-compose logs postgres

# Restart database
docker-compose restart postgres
```

**CLIP embeddings unavailable**
```bash
# Check PyTorch installation
uv run python -c "import torch; print(torch.__version__)"

# Reinstall CLIP (already in pyproject.toml)
uv sync --reinstall
```

**Vision API not working**
```bash
# Check Martian router
curl http://localhost:5333/health

# Restart router
docker-compose restart martian
```

### Performance Issues

**Slow captures**
- Reduce screenshot frequency
- Disable CLIP embeddings for speed
- Use smaller CLIP model (ViT-B/16)

**High memory usage**
- Reduce CLIP batch size
- Clean up old events regularly
- Use CPU-only PyTorch

## 🔒 Privacy & Security

- **Local-only**: No data leaves your machine
- **Encryption**: Optional AES encryption for database
- **Access control**: No network exposure by default
- **Data retention**: Automatic cleanup after 30 days
- **Audit**: Full logging of all operations

## 📈 Future Enhancements

- [ ] **Whisper integration** for voice queries
- [ ] **Timeline web UI** with React frontend
- [ ] **Mobile companion** app for remote search
- [ ] **Plugin system** for custom parsers
- [ ] **Federated sync** between devices
- [ ] **Advanced analytics** and insights

## 🤝 Contributing

This is a hackathon project (48-hour build). Contributions welcome!

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **OpenAI CLIP** for semantic understanding
- **Tesseract OCR** for text extraction
- **pgvector** for similarity search
- **Martian** for LLM routing
- **FastAPI** for the API framework
- **Rich** for beautiful CLI output

---

**Built with ❤️ in 48 hours for maximum productivity and zero privacy compromise.** 