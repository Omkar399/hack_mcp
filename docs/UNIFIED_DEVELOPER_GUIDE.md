# Eidolon AI Personal Assistant - Complete Developer Guide

This unified guide combines development setup, contribution guidelines, API documentation, and repository structure into a single comprehensive resource for developers.

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Development Setup](#development-setup)
3. [Contributing Guidelines](#contributing-guidelines)
4. [API Documentation](#api-documentation)
5. [Architecture Overview](#architecture-overview)
6. [Testing Strategy](#testing-strategy)
7. [Code Standards](#code-standards)

---

# Repository Structure

## 📁 Complete Directory Layout

```
eidolon/
├── 📄 README.md                          # Main project documentation
├── 📄 CLAUDE.md                          # Claude Code configuration
├── 📄 .gitignore                         # Git ignore patterns
├── 📄 pyproject.toml                     # Python project configuration
├── 📄 setup.py                           # Python package setup
├── 📄 requirements.txt                   # Production dependencies
├── 📄 requirements-dev.txt               # Development dependencies
├── 📄 validate_dependencies.py           # Dependency validation script
│
├── 🧪 test_phase1.py                     # Phase 1 validation tests
├── 🧪 test_phase2.py                     # Phase 2 validation tests
├── 🧪 test_phase3.py                     # Phase 3 validation tests
├── 🧪 test_phase4.py                     # Phase 4 validation tests
│
├── 📂 src/eidolon/                       # Main source code
│   ├── 📄 __init__.py                    # Package initialization
│   │
│   ├── 📂 cli/                           # Command-line interface
│   │   ├── 📄 __init__.py
│   │   └── 📄 main.py                    # CLI entry point
│   │
│   ├── 📂 core/                          # Core system components
│   │   ├── 📄 __init__.py
│   │   ├── 📄 observer.py                # Screenshot capture & monitoring
│   │   ├── 📄 analyzer.py                # OCR & AI content analysis
│   │   ├── 📄 memory.py                  # Memory management with NLP & RAG
│   │   └── 📄 interface.py               # User interface components
│   │
│   ├── 📂 models/                        # AI model integrations
│   │   ├── 📄 __init__.py
│   │   ├── 📄 local_vision.py            # Florence-2 vision model
│   │   ├── 📄 cloud_api.py               # Cloud AI APIs (Gemini, Claude, OpenAI)
│   │   └── 📄 decision_engine.py         # Intelligent routing logic
│   │
│   ├── 📂 storage/                       # Data storage & management
│   │   ├── 📄 __init__.py
│   │   ├── 📄 metadata_db.py             # SQLite database with FTS5
│   │   ├── 📄 vector_db.py               # ChromaDB vector database
│   │   └── 📄 file_manager.py            # File system operations
│   │
│   └── 📂 utils/                         # Shared utilities
│       ├── 📄 __init__.py
│       ├── 📄 config.py                  # Configuration management
│       ├── 📄 logging.py                 # Logging utilities
│       └── 📄 monitoring.py              # Performance monitoring
│
├── 📂 tests/                             # Test suite
│   ├── 📄 __init__.py
│   ├── 📄 test_observer.py               # Observer tests
│   ├── 📄 test_analyzer.py               # Analyzer tests
│   ├── 📄 test_memory.py                 # Memory system tests
│   └── 📄 test_integration.py            # Integration tests
│
├── 📂 config/                            # Configuration files
│   ├── 📄 settings.yaml                  # Main configuration
│   ├── 📄 settings-high-performance.yaml # High-performance template
│   └── 📄 logging.yaml                   # Logging configuration
│
├── 📂 docs/                              # Documentation
│   ├── 📄 README.md                      # Documentation index
│   ├── 📄 UNIFIED_USER_GUIDE.md          # Complete user guide
│   ├── 📄 UNIFIED_DEVELOPER_GUIDE.md     # This file
│   ├── 📂 api/                           # API documentation
│   ├── 📂 development/                   # Development docs
│   ├── 📂 examples/                      # Usage examples
│   └── 📂 user-guide/                    # User guides
│
├── 📂 data/                              # Generated data (gitignored)
│   ├── 📂 screenshots/                   # Captured screenshots
│   ├── 📂 vector_db/                     # ChromaDB storage
│   └── 📄 eidolon.db                     # SQLite database
│
└── 📂 logs/                              # Log files (gitignored)
    └── 📄 eidolon.log                    # Application logs
```

---

# Development Setup

## Prerequisites

- Python 3.9 or higher
- Git
- At least 8GB RAM (16GB recommended)
- 20GB free disk space
- Virtual environment tool

## Complete Setup Process

### 1. Fork and Clone

```bash
# Fork on GitHub first, then:
git clone https://github.com/YOUR_USERNAME/eidolon.git
cd eidolon

# Add upstream remote
git remote add upstream https://github.com/eidolon-ai/eidolon.git
```

### 2. Development Environment

```bash
# Create virtual environment
python3 -m venv eidolon_env
source eidolon_env/bin/activate  # Windows: eidolon_env\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install package in development mode
pip install -e .

# Install pre-commit hooks
pre-commit install

# Validate setup
python validate_dependencies.py
```

### 3. Configure Development Settings

```yaml
# config/settings-dev.yaml
observer:
  capture_interval: 5        # Faster for testing
  storage_path: "./test_data/screenshots"
  
logging:
  level: DEBUG              # More verbose logging
  
analysis:
  local_first: true         # Prefer local models for testing
```

### 4. Set Up IDE

#### VS Code
```json
// .vscode/settings.json
{
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.unittestEnabled": false
}
```

#### PyCharm
- Set interpreter to virtual environment
- Enable pytest as test runner
- Configure Black as formatter

---

# Contributing Guidelines

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Follow project standards

## Contribution Process

### 1. Find or Create Issue

```bash
# Check existing issues
# https://github.com/eidolon-ai/eidolon/issues

# Discuss new features
# https://github.com/eidolon-ai/eidolon/discussions
```

### 2. Create Feature Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-number-description
```

### 3. Development Workflow

```bash
# Make changes following TDD
# 1. Write test
# 2. Run test (should fail)
# 3. Implement feature
# 4. Run test (should pass)
# 5. Refactor if needed

# Run tests frequently
pytest tests/test_your_feature.py

# Check code quality
black src/
flake8 src/
mypy src/
```

### 4. Commit Guidelines

```bash
# Commit format
git commit -m "type: brief description

Detailed explanation of what and why (not how).

Fixes #123"

# Types: feat, fix, docs, style, refactor, test, chore
```

### 5. Submit Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create PR on GitHub
# - Clear title and description
# - Link to issue
# - Include test results
# - Add screenshots if UI changes
```

## Review Process

1. Automated checks must pass
2. Code review by maintainer
3. Address feedback
4. Approval and merge

---

# API Documentation

## Phase 4: Cloud AI & Semantic Memory APIs

### Vector Database API

```python
from eidolon.storage.vector_db import VectorDatabase, EmbeddingGenerator

# Initialize
vector_db = VectorDatabase()
embedding_gen = EmbeddingGenerator()

# Generate embeddings
embedding = embedding_gen.generate_text_embedding("sample text")
# Returns: List[float] with 384 dimensions

# Store content
success = vector_db.store_content(
    screenshot_id="screenshot_123",
    content_analysis={
        "content_type": "document",
        "description": "Python code example",
        "tags": ["python", "programming"]
    },
    extracted_text="def hello_world():\n    print('Hello')"
)

# Semantic search
results = vector_db.semantic_search(
    query="Python programming",
    n_results=10,
    content_type_filter="document",
    min_confidence=0.7
)

# Hybrid search (semantic + keyword)
results = vector_db.hybrid_search(
    query="machine learning",
    n_results=5,
    semantic_weight=0.7,
    keyword_weight=0.3
)
```

### Cloud AI Manager API

```python
from eidolon.models.cloud_api import CloudAPIManager
import asyncio

async def analyze_content():
    api_manager = CloudAPIManager()
    
    # Check available providers
    providers = api_manager.get_available_providers()
    # Returns: ["gemini", "claude", "openai"] if API keys set
    
    # Analyze image
    response = await api_manager.analyze_image(
        image_path="screenshot.png",
        prompt="What is the user working on?",
        preferred_provider="gemini",
        fallback=True  # Try other providers if preferred fails
    )
    
    # Analyze text
    response = await api_manager.analyze_text(
        text="Complex technical document...",
        analysis_type="summary",
        preferred_provider="claude"
    )
    
    # Get usage statistics
    stats = api_manager.get_usage_stats()
    # Returns: {"total_requests": 10, "total_cost": 0.25, "by_provider": {...}}

asyncio.run(analyze_content())
```

### Decision Engine API

```python
from eidolon.models.decision_engine import DecisionEngine, AnalysisRequest

# Initialize
engine = DecisionEngine()

# Create analysis request
request = AnalysisRequest(
    content_type="code",
    text_length=1000,
    has_image=True,
    importance=0.8,
    quality_requirement=0.9,
    user_query="Analyze this code for bugs"
)

# Get routing decision
decision = engine.make_routing_decision(
    request,
    available_providers=["gemini", "claude"]
)

print(f"Use cloud: {decision.use_cloud}")
print(f"Provider: {decision.provider}")
print(f"Reasoning: {decision.reasoning}")
print(f"Estimated cost: ${decision.estimated_cost}")
```

### Memory System API

```python
from eidolon.core.memory import MemorySystem
import asyncio

async def process_queries():
    memory = MemorySystem()
    
    # Natural language query
    response = await memory.process_natural_language_query(
        "Find all Python code from yesterday"
    )
    
    print(f"Intent: {response.intent}")  # "search"
    print(f"Results: {len(response.results)}")
    print(f"Response: {response.response}")
    
    # Parse query intent
    intent = memory.parse_query_intent("Summarize my work today")
    # Returns: QueryIntent(type="summarize", confidence=0.9, ...)
    
    # Parse time expressions
    time_range = memory.parse_time_expressions("from yesterday to today")
    # Returns: {"start": datetime(...), "end": datetime(...)}
    
    # Generate RAG response
    rag_response = await memory.generate_rag_response(
        query="What bugs did I fix?",
        search_results=response.results
    )

asyncio.run(process_queries())
```

## Core Component APIs

### Observer API

```python
from eidolon.core.observer import Observer

# Initialize
observer = Observer()

# Start/stop monitoring
observer.start_monitoring()
# ... monitoring happens ...
observer.stop_monitoring()

# Get status
status = observer.get_status()
# Returns: {
#   "running": True,
#   "capture_count": 42,
#   "last_capture": "2024-01-01T12:00:00",
#   "performance_metrics": {...}
# }

# Manual capture
screenshot = observer.capture_screenshot()
# Returns: Screenshot object with image data and metadata
```

### Analyzer API

```python
from eidolon.core.analyzer import Analyzer
from pathlib import Path

# Initialize
analyzer = Analyzer()

# Extract text (OCR)
result = analyzer.extract_text(Path("screenshot.png"))
# Returns: OCRResult(text="...", confidence=0.95, regions=[...])

# Analyze content
analysis = analyzer.analyze_content(Path("screenshot.png"))
# Returns: ContentAnalysis(
#   content_type="document",
#   description="...",
#   confidence=0.9,
#   tags=["python", "code"],
#   vision_analysis=VisionAnalysis(...)
# )

# Check AI availability
print(f"Tesseract: {analyzer._tesseract_available}")
print(f"Florence-2: {analyzer._florence_available}")
```

---

# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Interface Layer                        │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │     CLI     │  │   REST API   │  │   Web UI      │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                    Memory Layer                           │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  Vector DB  │  │   SQLite     │  │   NLP/RAG     │  │
│  │ (ChromaDB)  │  │   (FTS5)     │  │   Engine      │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                      AI Layer                             │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  Local AI   │  │  Cloud APIs  │  │   Decision    │  │
│  │(Florence-2) │  │(Gemini/etc) │  │   Engine      │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                   Analysis Layer                          │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │     OCR     │  │   Content    │  │    Scene      │  │
│  │ (Tesseract) │  │ Classification│ │ Understanding │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                   Observer Layer                          │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ Screenshot  │  │   Change     │  │  Performance  │  │
│  │  Capture    │  │  Detection   │  │  Monitoring   │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Data Flow

1. **Capture**: Observer takes screenshot
2. **Analysis**: Analyzer extracts text and classifies content
3. **AI Processing**: Local/cloud AI enhances understanding
4. **Storage**: Data stored in SQLite and vector DB
5. **Query**: User searches via CLI/API
6. **Retrieval**: Memory system processes query
7. **Response**: Results returned with optional RAG

## Key Design Patterns

### 1. Strategy Pattern
- Cloud API providers
- OCR engines
- Embedding models

### 2. Factory Pattern
- Screenshot creation
- Analysis result creation
- Response generation

### 3. Observer Pattern
- Performance monitoring
- Event notifications
- Status updates

### 4. Singleton Pattern
- Configuration management
- Logger instances
- Database connections

---

# Testing Strategy

## Test Organization

```
tests/
├── unit/              # Component tests
├── integration/       # System tests
├── performance/       # Benchmark tests
└── fixtures/          # Test data
```

## Testing Approach

### 1. Unit Tests

```python
# tests/unit/test_embedding_generator.py
import pytest
from eidolon.storage.vector_db import EmbeddingGenerator

class TestEmbeddingGenerator:
    def test_text_embedding_generation(self):
        gen = EmbeddingGenerator()
        embedding = gen.generate_text_embedding("test text")
        
        assert embedding is not None
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)
    
    def test_empty_text_handling(self):
        gen = EmbeddingGenerator()
        embedding = gen.generate_text_embedding("")
        
        assert embedding is None
```

### 2. Integration Tests

```python
# tests/integration/test_capture_to_search.py
import pytest
import asyncio
from eidolon.core.observer import Observer
from eidolon.core.memory import MemorySystem

@pytest.mark.integration
async def test_capture_to_search_workflow():
    # Capture
    observer = Observer()
    screenshot = observer.capture_screenshot()
    
    # Process
    # ... analysis steps ...
    
    # Search
    memory = MemorySystem()
    response = await memory.process_natural_language_query(
        "Find the latest screenshot"
    )
    
    assert len(response.results) > 0
```

### 3. Phase Tests

```python
# test_phase4.py structure
def test_phase4():
    """Comprehensive Phase 4 validation."""
    
    # 1. Vector Database
    test_vector_db_initialization()
    
    # 2. Cloud APIs
    test_cloud_api_integration()
    
    # 3. Decision Engine
    test_routing_logic()
    
    # 4. Semantic Search
    test_semantic_search()
    
    # 5. NLP
    test_natural_language_processing()
    
    # 6. Memory Integration
    test_memory_system()
    
    # 7. RAG
    test_rag_system()
```

## Test Commands

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/eidolon --cov-report=html

# Run specific test file
pytest tests/unit/test_vector_db.py

# Run integration tests only
pytest -m integration

# Run with verbose output
pytest -v

# Run phase validation
python test_phase1.py
python test_phase2.py
python test_phase3.py
python test_phase4.py
```

## Coverage Goals

- Unit tests: >90%
- Integration tests: >80%
- Overall: >85%

---

# Code Standards

## Python Style Guide

### General Rules

```python
# Good: Clear, explicit imports
from eidolon.storage.vector_db import VectorDatabase
from eidolon.core.memory import MemorySystem

# Bad: Star imports
from eidolon.storage import *

# Good: Type hints
def process_query(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Process a search query."""
    pass

# Bad: No type hints
def process_query(query, limit=10):
    pass
```

### Naming Conventions

```python
# Classes: PascalCase
class VectorDatabase:
    pass

# Functions/methods: snake_case
def semantic_search():
    pass

# Constants: UPPER_SNAKE_CASE
MAX_RESULTS = 100

# Private: Leading underscore
def _internal_method():
    pass
```

### Docstrings

```python
def semantic_search(
    self,
    query: str,
    n_results: int = 10,
    content_type_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Perform semantic search on stored content.
    
    Args:
        query: Search query text
        n_results: Maximum number of results to return
        content_type_filter: Optional filter by content type
        
    Returns:
        List of search results with similarity scores
        
    Raises:
        VectorDatabaseException: If search fails
        
    Example:
        >>> db = VectorDatabase()
        >>> results = db.semantic_search("Python code", n_results=5)
        >>> print(f"Found {len(results)} matches")
    """
    pass
```

### Error Handling

```python
# Good: Specific exceptions with context
try:
    result = vector_db.semantic_search(query)
except VectorDatabaseException as e:
    logger.error(f"Search failed for query '{query}': {e}")
    raise SearchException(f"Unable to search: {e}") from e

# Bad: Bare except
try:
    result = vector_db.semantic_search(query)
except:
    return None
```

### Async/Await

```python
# Good: Proper async usage
async def analyze_with_cloud(image_path: Path) -> CloudAPIResponse:
    """Analyze image using cloud AI."""
    async with CloudAPIManager() as manager:
        return await manager.analyze_image(image_path)

# Bad: Blocking in async
async def bad_analyze(image_path: Path):
    time.sleep(1)  # Don't block!
    return result
```

## Git Workflow

### Branch Naming

```bash
feature/add-semantic-search
fix/issue-123-memory-leak
docs/update-api-reference
refactor/optimize-embeddings
test/add-integration-tests
```

### Commit Messages

```bash
# Format
type(scope): subject

body

footer

# Examples
feat(vector-db): add hybrid search capability

Implement combined semantic and keyword search with configurable
weights. This allows users to get better results by combining
both search methods.

Closes #456

fix(memory): prevent duplicate storage in vector DB

Check for existing entries before storing to avoid duplicates.
This was causing inflated search results.

Fixes #789
```

## Code Review Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Type hints added
- [ ] Docstrings complete
- [ ] No hardcoded values
- [ ] Error handling appropriate
- [ ] Performance considered
- [ ] Security reviewed
- [ ] Breaking changes noted

---

## 🚀 Quick Reference

### Common Tasks

```bash
# Add new dependency
echo "new-package>=1.0.0" >> requirements.txt
pip install -r requirements.txt

# Run specific test
pytest tests/unit/test_new_feature.py -v

# Check code quality
black src/ && flake8 src/ && mypy src/

# Update documentation
# Edit relevant .md files
# Update docstrings
# Regenerate API docs if needed

# Create release
git tag -a v0.2.0 -m "Release version 0.2.0"
git push origin v0.2.0
```

### Debugging

```python
# Enable debug logging
import logging
logging.getLogger("eidolon").setLevel(logging.DEBUG)

# Use debugger
import pdb; pdb.set_trace()

# Profile performance
from eidolon.utils.monitoring import profile_performance

@profile_performance
def slow_function():
    pass
```

---

*Last updated: 2025-07-19 | Version: 0.1.0 | Phases 1-4 Complete*