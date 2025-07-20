# Screen Memory Assistant

A sophisticated screen capture and memory system that automatically captures, analyzes, and stores screen content with OCR, visual analysis, and semantic search capabilities. Built with FastAPI, PostgreSQL, and modern AI models.

## 🎯 Project Overview

This system automatically captures screenshots, extracts text via OCR, analyzes visual content using CLIP embeddings, and stores everything in a PostgreSQL database with vector similarity search. It's designed to create a comprehensive memory of your digital activities for future context-aware interactions.

## 🏗️ Architecture

### Core Components

- **Screen Capture**: `pyautogui` for screenshot capture with window information
- **Text Extraction**: EasyOCR for primary OCR, GPT-4o Vision via OpenRouter as fallback
- **Visual Analysis**: CLIP embeddings for semantic image understanding
- **Database**: PostgreSQL with `pgvector` extension for vector similarity search
- **API Server**: FastAPI with async processing and MCP-compatible endpoints
- **CLI Interface**: Command-line tool for database interactions
- **macOS Integration**: Keyboard shortcuts, notifications, and native UI integration

### Data Flow

1. **Capture Trigger**: Keyboard shortcut or API call initiates capture
2. **Screen Analysis**: OCR extracts text, CLIP generates visual embeddings
3. **Context Enrichment**: Window titles, timestamps, and metadata added
4. **Storage**: Data saved to PostgreSQL with vector embeddings
5. **Search**: Semantic similarity search across text and visual content

## 🚀 Current Features

### ✅ Implemented

- **Automatic Screen Capture**: Trigger via keyboard shortcuts or API
- **Multi-Modal Analysis**: OCR + visual embeddings for comprehensive understanding
- **Async Processing**: Non-blocking capture with background task processing
- **macOS Integration**: Native notifications, keyboard shortcuts, accessibility support
- **Vector Database**: PostgreSQL with semantic search capabilities
- **API Server**: RESTful endpoints for capture and search operations
- **CLI Tool**: Command-line interface for database queries
- **Environment Management**: Secure API key handling with `python-dotenv`
- **Docker Support**: Containerized PostgreSQL with pgvector

### 🎯 Key Capabilities

- **Instant Capture**: Sub-second screenshot capture with immediate feedback
- **Smart Fallbacks**: OCR → Vision API fallback for better text extraction
- **Context Preservation**: Window titles, timestamps, and application context
- **Semantic Search**: Find relevant screenshots by text or visual similarity
- **Concurrent Processing**: Multiple captures can be processed simultaneously
- **macOS Native**: Toast notifications, keyboard shortcuts, accessibility permissions

## 📋 Setup Instructions

### Prerequisites

- macOS (for native integration features)
- Python 3.8+
- Docker and Docker Compose
- OpenRouter API key

### Installation

1. **Clone and Setup**:
   ```bash
   git clone <repository-url>
   cd hack_mcp
   ```

2. **Environment Configuration**:
   ```bash
   cp .env.example .env
   # Edit .env with your OpenRouter API key
   ```

3. **Install Dependencies**:
   ```bash
   uv sync
   ```

4. **Start Database**:
   ```bash
   docker-compose up -d
   ```

5. **Initialize Database**:
   ```bash
   python -c "from database import init_db; import asyncio; asyncio.run(init_db())"
   ```

### macOS Shortcut Setup

1. **Create Shortcut**:
   - Open macOS Shortcuts app
   - Create new shortcut
   - Add "Run Shell Script" action
   - Set script to: `/bin/zsh /path/to/hack_mcp/capture_shortcut.sh`
   - Save as "Screen Capture"

2. **Set Keyboard Shortcut**:
   - System Preferences → Keyboard → Shortcuts
   - Add shortcut for your Shortcuts app shortcut
   - Recommended: `Cmd+Shift+S`

3. **Accessibility Permissions**:
   - System Preferences → Security & Privacy → Privacy → Accessibility
   - Add Terminal and Shortcuts app

## 🛠️ Usage

### Starting the System

```bash
# Start API server and hotkey daemon
./start_with_hotkeys.sh

# Or start components individually
python screen_api.py  # API server
python hotkey_daemon.py  # Hotkey listener
```

### Capture Methods

1. **Keyboard Shortcut**: Press your configured shortcut (e.g., `Cmd+Shift+S`)
2. **API Call**: `curl -X POST http://localhost:8000/capture`
3. **CLI Tool**: `python cli.py capture`

### Search and Query

```bash
# Search by text
python cli.py search "find screenshots with login forms"

# Search by similarity
python cli.py similar <screenshot_id>

# List recent captures
python cli.py list --limit 10
```

### API Endpoints

- `POST /capture` - Trigger screen capture
- `GET /captures` - List recent captures
- `GET /captures/{id}` - Get specific capture details
- `GET /search` - Semantic search across captures
- `GET /status` - Capture processing status

## 📁 Project Structure

```
hack_mcp/
├── capture.py              # Core capture logic with OCR and CLIP
├── screen_api.py           # FastAPI server with async processing
├── database.py             # SQLAlchemy async database operations
├── cli.py                  # Command-line interface
├── hotkey_daemon.py        # Keyboard shortcut listener
├── simple_capture.py       # Standalone capture script
├── capture_shortcut.sh     # macOS Shortcuts integration
├── instant_capture.sh      # Fast capture script
├── reliable_capture.sh     # Reliable capture script
├── start_with_hotkeys.sh   # System startup script
├── docker-compose.yml      # PostgreSQL container setup
├── pyproject.toml          # Python dependencies
└── README.md              # This file
```

## 🔧 Configuration

### Environment Variables

```bash
# Required
OPENROUTER_API_KEY=your_api_key_here

# Optional
DATABASE_URL=postgresql://user:pass@localhost/screen_memory
LOG_LEVEL=INFO
```

### Database Schema

- `screen_events`: Main capture data with text, embeddings, metadata
- `capture_status`: Processing status tracking
- Vector similarity search via pgvector extension

## 🎯 Next Steps: EnrichMCP Integration & Popup Chat Bot

### Phase 1: EnrichMCP Integration

**Goal**: Implement standardized MCP (Model Context Protocol) compatibility for better AI model integration and context management.

**Implementation Plan**:
1. **MCP Server Setup**:
   - Create MCP-compatible server endpoints
   - Implement standard MCP resource and tool protocols
   - Add context retrieval and search capabilities

2. **Enhanced Context Management**:
   - Structured context formatting for AI models
   - Temporal context windows (recent vs. historical)
   - Multi-modal context aggregation (text + visual)

3. **Standardized API**:
   - MCP-compatible resource endpoints
   - Tool definitions for capture, search, and analysis
   - Context streaming for real-time updates

### Phase 2: Popup Chat Bot

**Goal**: Create an intelligent chat interface that can answer questions about captured screen content using both text and visual context.

**Implementation Plan**:
1. **Chat Interface**:
   - Native macOS popup window
   - Real-time chat with typing indicators
   - Keyboard shortcut activation (`Cmd+Shift+C`)

2. **Context-Aware Responses**:
   - Query understanding and intent classification
   - Multi-modal context retrieval (text + visual similarity)
   - Temporal relevance scoring

3. **AI Integration**:
   - OpenRouter integration for chat responses
   - Context-aware prompt engineering
   - Response generation with source attribution

4. **Advanced Features**:
   - Follow-up question handling
   - Context memory across conversation
   - Action suggestions based on screen content

### Technical Architecture for Chat Bot

```
User Query → Intent Classification → Context Retrieval → Response Generation
     ↓              ↓                    ↓                    ↓
Chat Interface → Query Parser → Multi-Modal Search → AI Model + Context
```

### Implementation Details

1. **Chat Bot Components**:
   - `chat_bot.py`: Core chat logic and AI integration
   - `chat_ui.py`: Native macOS popup interface
   - `context_engine.py`: Multi-modal context retrieval
   - `intent_classifier.py`: Query understanding and routing

2. **MCP Integration**:
   - `mcp_server.py`: MCP-compatible server implementation
   - `mcp_resources.py`: Resource definitions and handlers
   - `mcp_tools.py`: Tool implementations for capture and search

3. **Enhanced Database**:
   - Conversation history storage
   - Context relevance scoring
   - Query-answer pairs for learning

### Expected User Experience

1. **Capture**: `Cmd+Shift+S` captures current screen
2. **Chat**: `Cmd+Shift+C` opens popup chat bot
3. **Query**: "What was I working on 10 minutes ago?"
4. **Response**: AI provides context-aware answer with relevant screenshots
5. **Follow-up**: "Show me the login form I was filling out"
6. **Action**: System retrieves and displays relevant capture

## 🐛 Troubleshooting

### Common Issues

1. **Permission Denied**: Ensure Terminal and Shortcuts have accessibility permissions
2. **Database Connection**: Check Docker is running and database is initialized
3. **API Key**: Verify OpenRouter API key is set in `.env`
4. **Hotkey Not Working**: Check macOS Shortcuts setup and keyboard shortcut assignment

### Debug Mode

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
python screen_api.py

# Check capture status
curl http://localhost:8000/status
```

## 🤝 Contributing

This project is actively developed. Key areas for contribution:
- EnrichMCP integration
- Chat bot implementation
- UI/UX improvements
- Performance optimization
- Additional AI model integrations

## 📄 License

[Add your license information here]

---

**Current Status**: ✅ Core system operational with capture, storage, and search capabilities
**Next Milestone**: 🚧 EnrichMCP integration and popup chat bot implementation 