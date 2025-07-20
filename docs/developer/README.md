# Eidolon Developer Documentation

Welcome to the comprehensive developer documentation for Eidolon AI Personal Assistant. This guide covers everything you need to contribute to, extend, or integrate with Eidolon.

## 📚 Table of Contents

### Getting Started
- **[Development Setup](setup.md)** - Set up your development environment
- **[Project Architecture](architecture.md)** - Understanding Eidolon's structure
- **[Contributing Guidelines](contributing.md)** - How to contribute to the project
- **[Code Standards](code-standards.md)** - Coding conventions and best practices

### Core Development
- **[API Reference](api-reference.md)** - Complete API documentation
- **[Database Schema](database.md)** - Database structure and relationships
- **[Testing Guide](testing.md)** - Testing strategies and frameworks
- **[Building & Packaging](building.md)** - Build processes and distribution

### AI & ML Development
- **[Vision Models](ai/vision-models.md)** - Local AI model integration
- **[Cloud AI APIs](ai/cloud-apis.md)** - Cloud service integrations
- **[Embeddings & Vector DB](ai/embeddings.md)** - Semantic search implementation
- **[Decision Engine](ai/decision-engine.md)** - Local vs cloud routing logic

### Extension Development
- **[Plugin Architecture](plugins/architecture.md)** - Plugin system overview
- **[Creating Plugins](plugins/creating.md)** - Step-by-step plugin development
- **[Plugin API](plugins/api.md)** - Plugin development APIs
- **[Example Plugins](plugins/examples.md)** - Sample plugin implementations

### Integration Development
- **[MCP Integration](integration/mcp.md)** - Model Context Protocol implementation
- **[REST API](integration/rest-api.md)** - HTTP API for external integrations
- **[WebSocket API](integration/websocket.md)** - Real-time communication
- **[CLI Extensions](integration/cli.md)** - Command-line interface extensions

### Advanced Topics
- **[Performance Optimization](advanced/performance.md)** - Optimization strategies
- **[Security Implementation](advanced/security.md)** - Security features and protocols
- **[Data Pipeline](advanced/data-pipeline.md)** - Data processing architecture
- **[Monitoring & Metrics](advanced/monitoring.md)** - System monitoring implementation

### DevOps & Deployment
- **[Development Workflow](devops/workflow.md)** - Git workflow and CI/CD
- **[Docker Development](devops/docker.md)** - Containerized development
- **[Release Process](devops/releases.md)** - Release management
- **[Debugging Guide](devops/debugging.md)** - Debugging techniques and tools

## 🚀 Quick Start for Developers

### 1. Set Up Development Environment

```bash
# Clone the repository
git clone https://github.com/eidolon-ai/eidolon.git
cd eidolon

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt
pip install -e .

# Install pre-commit hooks
pre-commit install
```

### 2. Run Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=eidolon --cov-report=html

# Run specific test suite
python -m pytest tests/unit/
python -m pytest tests/integration/
```

### 3. Start Development Server

```bash
# Start with hot reloading
python -m eidolon serve --reload --debug

# Run background processes
python -m eidolon capture --debug
```

## 🏗️ Architecture Overview

Eidolon follows a modular, layered architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    Interface Layer                          │
│  CLI • REST API • WebSocket • Chat • Web UI                │
├─────────────────────────────────────────────────────────────┤
│                     Core Layer                              │
│  Observer • Analyzer • Memory • Query Processor             │
├─────────────────────────────────────────────────────────────┤
│                    Models Layer                             │
│  Local Vision • Cloud APIs • Decision Engine                │
├─────────────────────────────────────────────────────────────┤
│                    Storage Layer                            │
│  Vector DB • SQLite • File Manager                         │
├─────────────────────────────────────────────────────────────┤
│                    Utils Layer                              │
│  Config • Logging • Monitoring • Production                │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Observer**: Screenshot capture and change detection
2. **Analyzer**: Content analysis using AI models
3. **Memory**: Vector database and semantic search
4. **Interface**: User interaction layer (CLI, API, Chat)
5. **Models**: AI model management and routing
6. **Storage**: Data persistence and file management

## 🔧 Development Stack

### Core Technologies
- **Python 3.9+**: Primary language
- **FastAPI**: Web framework for REST APIs
- **SQLite**: Metadata storage
- **ChromaDB**: Vector database for embeddings
- **Transformers**: Local AI model framework

### AI/ML Stack
- **Florence-2**: Vision understanding model
- **CLIP**: Content classification model
- **Sentence Transformers**: Text embeddings
- **Tesseract/EasyOCR**: OCR engines
- **OpenAI/Claude/Gemini**: Cloud AI APIs

### Development Tools
- **pytest**: Testing framework
- **black**: Code formatting
- **flake8**: Linting
- **mypy**: Type checking
- **pre-commit**: Git hooks

### Build & Distribution
- **setuptools**: Package building
- **Docker**: Containerization
- **GitHub Actions**: CI/CD
- **PyPI**: Package distribution

## 🎯 Development Phases

### Current Status (Phase 4 Complete)
- ✅ **Phase 1**: Foundation & Screenshot Capture
- ✅ **Phase 2**: Intelligent Capture & OCR
- ✅ **Phase 3**: Local AI Integration
- ✅ **Phase 4**: Cloud AI & Semantic Memory
- 🚧 **Phase 5**: Advanced Analytics (In Progress)
- 📋 **Phase 6**: MCP Integration & Basic Agency
- 📋 **Phase 7**: Advanced Agency & Digital Twin

### Ongoing Development Areas

#### High Priority
- Advanced analytics and insights
- MCP server implementation
- Performance optimization
- Security enhancements

#### Medium Priority
- Plugin architecture
- Mobile integration
- Real-time collaboration
- Advanced automation

#### Future Features
- Federated learning
- Multi-modal understanding
- Predictive assistance
- Enterprise features

## 🔍 Contributing Areas

### Code Contributions
- **Core Features**: Enhance existing components
- **AI Models**: Integrate new models or improve existing ones
- **Performance**: Optimize algorithms and data structures
- **Security**: Strengthen security and privacy features

### Testing & Quality
- **Unit Tests**: Increase test coverage
- **Integration Tests**: End-to-end testing scenarios
- **Performance Tests**: Benchmarking and optimization
- **Security Tests**: Vulnerability assessment

### Documentation
- **API Documentation**: Complete API reference
- **Tutorials**: User and developer tutorials
- **Architecture Docs**: System design documentation
- **Examples**: Code samples and use cases

### DevOps & Infrastructure
- **CI/CD**: Improve build and deployment processes
- **Docker**: Container optimization
- **Monitoring**: Observability and metrics
- **Packaging**: Distribution improvements

## 📋 Development Guidelines

### Code Quality Standards
- **Type Hints**: All functions must have type annotations
- **Documentation**: Comprehensive docstrings using Google style
- **Testing**: Minimum 90% test coverage for new code
- **Security**: Security-first design principles

### Git Workflow
- **Feature Branches**: Use feature branches for development
- **Pull Requests**: All changes via reviewed pull requests
- **Conventional Commits**: Follow conventional commit format
- **Linear History**: Rebase for clean history

### Review Process
- **Code Review**: All PRs require review by maintainers
- **Automated Checks**: CI must pass before merge
- **Documentation**: Update docs with code changes
- **Backward Compatibility**: Maintain API compatibility

## 🛠️ Development Tools Setup

### IDE Configuration

#### VS Code
```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true
}
```

#### PyCharm
- Enable type checking with mypy
- Configure black formatter
- Set up pytest as test runner
- Enable pre-commit hooks

### Environment Configuration

```bash
# Development environment variables
export EIDOLON_ENV=development
export EIDOLON_DEBUG=true
export EIDOLON_LOG_LEVEL=DEBUG

# Test environment
export EIDOLON_TEST_DB=":memory:"
export EIDOLON_TEST_STORAGE="/tmp/eidolon-test"
```

## 📞 Developer Support

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General development discussions
- **Developer Discord**: Real-time chat (invite in repo)
- **Office Hours**: Weekly developer office hours

### Getting Help
1. Check existing [GitHub Issues](https://github.com/eidolon-ai/eidolon/issues)
2. Search [Developer Discussions](https://github.com/eidolon-ai/eidolon/discussions)
3. Join developer Discord for real-time help
4. Attend weekly office hours for complex questions

### Reporting Issues
When reporting development issues, include:
- Python version and OS
- Complete error traceback
- Steps to reproduce
- Expected vs actual behavior
- Relevant code snippets

## 🔒 Security for Developers

### Secure Development Practices
- **Input Validation**: Validate all inputs
- **Dependency Security**: Regular security updates
- **Secret Management**: Never commit secrets
- **Audit Trail**: Log security-relevant events

### Security Testing
- **SAST**: Static analysis with security tools
- **Dependency Scanning**: Check for vulnerable dependencies
- **Container Scanning**: Security scan Docker images
- **Penetration Testing**: Regular security assessments

---

Welcome to the Eidolon development community! We're excited to have you contribute to building the future of AI-powered personal assistance.