# Eidolon API Documentation

This directory contains API documentation for all components of the Eidolon AI Personal Assistant.

## 📚 Available API Documentation

### Phase-Specific APIs

- **[Phase 4 API](phase4-api.md)** - Cloud AI & Semantic Memory
  - Vector Database (ChromaDB)
  - Cloud AI Providers (Gemini, Claude, OpenAI)
  - Decision Engine
  - Natural Language Processing
  - RAG System

### Core Component APIs (Coming Soon)

- **Phase 1 API** - Observer System
  - Screenshot capture
  - Change detection
  - Performance monitoring
  
- **Phase 2 API** - Analysis System
  - OCR text extraction
  - Content classification
  - Database operations
  
- **Phase 3 API** - Local AI Integration
  - Florence-2 vision model
  - Scene classification
  - Object detection

## 🔧 API Design Principles

Our APIs follow these design principles:

1. **Consistency** - Similar patterns across all components
2. **Type Safety** - Full type hints and validation
3. **Error Handling** - Clear error messages and recovery
4. **Performance** - Async/await for non-blocking operations
5. **Documentation** - Comprehensive docstrings and examples

## 📝 API Usage Patterns

### Basic Pattern
```python
from eidolon.component import Component

# Initialize
component = Component(config)

# Use synchronous methods
result = component.method(params)

# Use async methods
result = await component.async_method(params)
```

### Error Handling
```python
try:
    result = component.risky_operation()
except ComponentException as e:
    logger.error(f"Operation failed: {e}")
    # Handle gracefully
```

### Configuration
```python
from eidolon.utils.config import get_config

config = get_config()
component = Component(config)
```

## 🚀 Quick Start Examples

### Vector Database Search
```python
from eidolon.storage.vector_db import VectorDatabase

vector_db = VectorDatabase()
results = vector_db.semantic_search("Python programming", n_results=5)
```

### Cloud AI Analysis
```python
from eidolon.models.cloud_api import CloudAPIManager

api_manager = CloudAPIManager()
response = await api_manager.analyze_text("Analyze this text", analysis_type="summary")
```

### Natural Language Query
```python
from eidolon.core.memory import MemorySystem

memory = MemorySystem()
response = await memory.process_natural_language_query("What did I work on yesterday?")
```

## 📊 API Categories

### Storage APIs
- File management
- Database operations
- Vector storage
- Metadata handling

### AI Model APIs
- Local models (Florence-2, CLIP)
- Cloud providers (Gemini, Claude, OpenAI)
- Embedding generation
- Decision routing

### Core System APIs
- Observer (monitoring)
- Analyzer (content analysis)
- Memory (knowledge base)
- Interface (user interaction)

### Utility APIs
- Configuration management
- Logging system
- Performance monitoring
- Error handling

## 🔐 Security Considerations

- API keys stored in environment variables
- No sensitive data in logs
- Input validation on all endpoints
- Rate limiting for cloud APIs
- Local-only mode available

## 📝 Contributing

To add new API documentation:
1. Create a new markdown file for your component
2. Follow the existing format and structure
3. Include complete examples
4. Document all parameters and return types
5. Add to this README's index

## 🆘 Need Help?

- See [Phase 4 API](phase4-api.md) for a complete example
- Check the [main documentation](../README.md)
- Visit [GitHub Issues](https://github.com/eidolon-ai/eidolon/issues) for support