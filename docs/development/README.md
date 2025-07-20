# Eidolon Development Documentation

This directory contains documentation for developers working on or contributing to the Eidolon AI Personal Assistant project.

## 📚 Development Documentation

### Core Documents

- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute to Eidolon
  - Development setup
  - Code standards
  - Pull request process
  - Testing requirements

- **[Progress Plan](PROGRESS_PLAN.md)** - Detailed development roadmap
  - Phase breakdown (1-7)
  - Task tracking
  - Timeline estimates
  - Current status

- **[Repository Structure](REPOSITORY_STRUCTURE.md)** - Project organization
  - Directory layout
  - Module descriptions
  - File purposes
  - Architecture overview

### Additional Resources (Coming Soon)

- **Architecture Guide** - System design and patterns
- **Testing Guide** - Comprehensive testing strategies
- **Performance Guide** - Optimization techniques
- **Security Guide** - Security best practices

## 🚀 Development Quick Start

### Setup Development Environment
```bash
# Clone the repository
git clone https://github.com/eidolon-ai/eidolon.git
cd eidolon

# Create virtual environment
python3 -m venv eidolon_env
source eidolon_env/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov

# Run specific phase tests
python test_phase1.py
python test_phase2.py
python test_phase3.py
python test_phase4.py
```

### Code Quality
```bash
# Format code
black src/

# Check linting
flake8 src/

# Type checking
mypy src/
```

## 📊 Project Status

### Completed Phases
- ✅ **Phase 1**: Observer System (Foundation)
- ✅ **Phase 2**: Analysis System (OCR & Classification)
- ✅ **Phase 3**: Local AI Integration (Florence-2)
- ✅ **Phase 4**: Cloud AI & Semantic Memory

### Upcoming Phases
- 📋 **Phase 5**: Advanced Analytics
- 📋 **Phase 6**: MCP Integration & Agency
- 📋 **Phase 7**: Digital Twin Capabilities

## 🏗️ Architecture Overview

```
Eidolon Architecture
├── Observer Layer      # Screenshot capture and monitoring
├── Analysis Layer      # OCR and content classification
├── AI Layer           # Local and cloud AI processing
├── Memory Layer       # Vector DB and knowledge base
└── Interface Layer    # CLI and API interfaces
```

### Key Technologies
- **Python 3.9+** - Core language
- **ChromaDB** - Vector database
- **Sentence Transformers** - Embeddings
- **Florence-2** - Vision AI
- **Cloud APIs** - Gemini, Claude, OpenAI
- **SQLite** - Metadata storage
- **FastAPI** - Web API (future)

## 🧪 Testing Strategy

### Test Levels
1. **Unit Tests** - Individual components
2. **Integration Tests** - Component interactions
3. **Phase Tests** - End-to-end validation
4. **Performance Tests** - Resource usage

### Test Coverage Goals
- Core components: >90%
- Utilities: >80%
- Overall: >85%

## 🔧 Development Tools

### Recommended IDE Setup
- **VS Code** with Python extension
- **PyCharm Professional**
- **GitHub Copilot** (optional)

### Essential Extensions
- Python language server
- Black formatter
- Flake8 linter
- GitLens
- Markdown preview

## 📝 Code Standards

### Python Style
- Follow PEP 8
- Use Black formatting
- Type hints required
- Docstrings for public APIs

### Git Workflow
- Feature branches
- Descriptive commits
- PR reviews required
- CI/CD checks must pass

### Documentation
- Update docs with code
- Include examples
- Keep README current
- Version all changes

## 🤝 Contributing Process

1. **Find an Issue**
   - Check open issues
   - Discuss new features
   - Get approval for major changes

2. **Development**
   - Fork the repository
   - Create feature branch
   - Write tests first (TDD)
   - Implement solution

3. **Quality Assurance**
   - Run all tests
   - Check code coverage
   - Update documentation
   - Verify no regressions

4. **Submit PR**
   - Clear description
   - Link to issue
   - Pass all checks
   - Address reviews

## 🚀 Deployment

### Local Development
```bash
pip install -e .
```

### Package Building
```bash
python -m build
```

### Testing Installation
```bash
pip install dist/eidolon-*.whl
```

## 🆘 Developer Support

- **Discord**: Developer chat (coming soon)
- **GitHub Discussions**: Technical questions
- **Issue Tracker**: Bug reports
- **Wiki**: Extended documentation

## 📈 Performance Considerations

- Memory usage targets
- CPU optimization
- Storage efficiency
- Network minimization
- Battery impact (laptops)

## 🔐 Security Development

- No hardcoded secrets
- Input validation
- Safe file operations
- API key management
- Privacy by design

Welcome to the Eidolon development team!