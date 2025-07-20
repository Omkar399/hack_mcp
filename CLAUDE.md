# Eidolon AI Personal Assistant - Technical Specifications

## Project Overview
Eidolon is a hyper-personalized AI assistant system that functions as a digital twin by monitoring computer activity, building comprehensive knowledge bases, and acting autonomously. The system provides perfect recall, contextual understanding, and proactive assistance while learning user patterns and communication styles.

## Build Commands
- `python -m pip install -e .`: Install project in development mode
- `python -m pytest`: Run the full test suite
- `python -m pytest --cov`: Run tests with coverage report
- `python -m eidolon --help`: Show CLI help
- `python -m eidolon capture`: Start screenshot capture
- `python -m eidolon search "<query>"`: Search captured content
- `python -m eidolon status`: Show system status

## Architecture Overview

### Core Components
1. **Observer Layer**: Silent monitoring and intelligent capture
2. **Analysis Layer**: Local triage and cloud deep analysis
3. **Memory Layer**: Vector knowledge base and temporal organization
4. **Interface Layer**: Natural language queries and agentic actions

### Technology Stack
- **Language**: Python 3.9+
- **Screenshot Capture**: mss, PIL (Pillow)
- **OCR**: Tesseract, EasyOCR
- **Local AI**: Florence-2, CLIP, Transformers
- **Cloud AI**: Gemini, Claude, GPT-4V APIs
- **Vector Database**: ChromaDB, FAISS
- **Traditional Database**: SQLite for metadata
- **Web Interface**: FastAPI, Streamlit
- **Testing**: pytest, pytest-cov
- **Package Management**: Poetry or pip

### Directory Structure
```
eidolon/
├── src/eidolon/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── observer.py          # Screenshot capture and monitoring
│   │   ├── analyzer.py          # AI analysis and content understanding
│   │   ├── memory.py            # Knowledge base and storage
│   │   └── interface.py         # CLI and API interfaces
│   ├── models/
│   │   ├── __init__.py
│   │   ├── local_vision.py      # Local AI models
│   │   ├── cloud_api.py         # Cloud AI integrations
│   │   └── decision_engine.py   # Local vs cloud routing
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── vector_db.py         # Vector database operations
│   │   ├── metadata_db.py       # SQLite metadata storage
│   │   └── file_manager.py      # File system operations
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py            # Configuration management
│   │   ├── logging.py           # Logging setup
│   │   └── monitoring.py        # Performance monitoring
│   └── cli/
│       ├── __init__.py
│       └── main.py              # CLI entry point
├── tests/
│   ├── __init__.py
│   ├── test_observer.py
│   ├── test_analyzer.py
│   ├── test_memory.py
│   └── test_integration.py
├── data/
│   ├── screenshots/             # Captured screenshots
│   ├── extracted/               # Processed content
│   └── models/                  # Local model storage
├── config/
│   ├── settings.yaml            # Main configuration
│   └── logging.yaml             # Logging configuration
├── docs/
│   ├── API.md                   # API documentation
│   ├── ARCHITECTURE.md          # Technical architecture
│   └── USER_GUIDE.md            # User documentation
├── pyproject.toml               # Project configuration
├── requirements.txt             # Dependencies
├── README.md                    # Project readme
├── CLAUDE.md                    # This file
└── PROGRESS_PLAN.md             # Development progress tracking
```

## Development Phases

### Phase 1: Foundation (Days 1-3)
**Core Infrastructure Setup**
- Project initialization with proper Python packaging
- Basic screenshot capture using mss library
- File storage and organization system
- Configuration management and logging
- Testing framework setup

### Phase 2: Intelligent Capture (Days 4-7)
**Smart Monitoring System**
- Change detection algorithms
- Activity-triggered capture
- OCR text extraction
- Content categorization
- Basic search functionality

### Phase 3: Local AI Integration (Days 8-12)
**Vision Model Integration**
- Florence-2 for image understanding
- CLIP for content classification
- UI element detection
- Importance scoring
- Enhanced search capabilities

### Phase 4: Cloud AI & Semantic Memory (Days 13-18)
**Advanced Analysis System**
- Tiered local/cloud analysis
- Vector database implementation
- Semantic search with embeddings
- Natural language query interface
- RAG (Retrieval-Augmented Generation)

### Phase 5: Advanced Memory & Analytics (Days 19-24)
**Deep Understanding System**
- Project timeline reconstruction
- Communication analysis
- Productivity insights
- Personal pattern learning
- Advanced query processing

### Phase 6: MCP Integration & Basic Agency (Days 25-30)
**Autonomous Action System**
- Model Context Protocol server
- Basic task automation
- Email and document assistance
- Tool orchestration
- Safety mechanisms

### Phase 7: Advanced Agency & Digital Twin (Days 31-40)
**Complete AI Assistant**
- Complex task planning
- Proactive assistance
- Style replication
- Full system integration
- Ecosystem orchestration

## Configuration Schema

### Main Settings (settings.yaml)
```yaml
observer:
  capture_interval: 10          # Seconds between screenshots
  activity_threshold: 0.05      # Change detection sensitivity
  storage_path: "./data/screenshots"
  max_storage_gb: 50           # Storage limit

analysis:
  local_models:
    vision: "microsoft/florence-2-base"
    clip: "openai/clip-vit-base-patch32"
  cloud_apis:
    gemini_key: "${GEMINI_API_KEY}"
    claude_key: "${CLAUDE_API_KEY}"
    openai_key: "${OPENAI_API_KEY}"
  routing:
    importance_threshold: 0.7   # Route to cloud if above threshold
    cost_limit_daily: 10.0     # Daily API cost limit

memory:
  vector_db: "chromadb"        # chromadb, faiss, or pinecone
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  chunk_size: 512
  overlap: 50

privacy:
  sensitive_patterns:          # Auto-redact these patterns
    - "password"
    - "api_key"
    - "ssn"
  excluded_apps:              # Don't monitor these apps
    - "1Password"
    - "Keychain Access"
  local_only_mode: false      # Never send to cloud APIs
```

## API Specifications

### Core Observer API
```python
class Observer:
    def start_monitoring(self) -> None
    def stop_monitoring(self) -> None
    def capture_screenshot(self) -> Screenshot
    def detect_changes(self, prev: Screenshot, curr: Screenshot) -> ChangeMetrics
    def extract_text(self, screenshot: Screenshot) -> ExtractedText
```

### Analysis Engine API
```python
class AnalysisEngine:
    def analyze_local(self, content: Content) -> LocalAnalysis
    def analyze_cloud(self, content: Content) -> CloudAnalysis
    def route_decision(self, content: Content) -> RoutingDecision
    def extract_insights(self, analysis: Analysis) -> Insights
```

### Memory System API
```python
class MemorySystem:
    def store(self, content: Content, metadata: Metadata) -> str
    def search(self, query: str, filters: Dict) -> List[SearchResult]
    def retrieve_context(self, id: str) -> Context
    def update_embeddings(self, content: Content) -> None
```

### Interface API
```python
class Interface:
    def process_query(self, query: str) -> Response
    def execute_action(self, action: Action) -> ActionResult
    def get_status(self) -> SystemStatus
    def export_data(self, format: str, filters: Dict) -> ExportResult
```

## Security and Privacy

### Data Protection
- All data encrypted at rest using AES-256
- Sensitive information automatically redacted
- User control over cloud data transmission
- Configurable data retention policies
- Export and deletion capabilities

### Access Control
- Local-first architecture by default
- Granular permissions for different data types
- API key management and rotation
- Audit logging for all data access
- Privacy compliance features (GDPR, CCPA)

## Performance Requirements

### Resource Constraints
- Memory usage: <500MB baseline, <2GB with full analysis
- CPU usage: <5% average, <20% during analysis bursts
- Storage growth: <1GB per week typical usage
- Network usage: <100MB per day for cloud analysis

### Response Times
- Screenshot capture: <100ms
- Local analysis: <2 seconds
- Cloud analysis: <10 seconds
- Search queries: <500ms
- Real-time monitoring: <50ms latency

## Testing Strategy

### Unit Testing
- Individual component testing with pytest
- Mock external dependencies (APIs, file system)
- Coverage target: >90% for core components
- Property-based testing for data transformations

### Integration Testing
- End-to-end workflow validation
- Database integration testing
- API integration testing
- Performance benchmarking

### User Acceptance Testing
- Real-world usage scenarios
- Privacy and security validation
- Cross-platform compatibility
- Accessibility compliance

## Quality Assurance

### Code Quality
- Type hints for all functions
- Docstrings following Google style
- Black code formatting
- Flake8 linting
- Pre-commit hooks

### Documentation
- Comprehensive API documentation
- User guides and tutorials
- Architecture decision records
- Troubleshooting guides

### Monitoring
- Performance metrics collection
- Error tracking and alerting
- User behavior analytics
- System health monitoring

## Deployment and Distribution

### Installation Methods
- PyPI package distribution
- Docker container images
- Desktop application bundles
- Cloud deployment options

### Platform Support
- Windows 10/11
- macOS 10.15+
- Linux (Ubuntu 20.04+)
- Cross-platform compatibility

### Update Mechanism
- Automatic update notifications
- Incremental model updates
- Configuration migration
- Rollback capabilities

## Future Enhancements

### Advanced Features
- Multi-monitor support
- Mobile device integration
- Team collaboration features
- Enterprise deployment
- Advanced analytics dashboard

### AI Capabilities
- Custom model fine-tuning
- Federated learning
- Advanced reasoning
- Multi-modal understanding
- Predictive assistance

### Integration Ecosystem
- Plugin architecture
- Third-party integrations
- API marketplace
- Community extensions
- Enterprise connectors

This specification provides the foundation for building Eidolon as a comprehensive AI personal assistant system with proper architecture, security, and scalability considerations.