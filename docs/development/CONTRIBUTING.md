# Contributing to Eidolon AI Personal Assistant

We welcome contributions to Eidolon! This guide outlines how to contribute effectively.

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- At least 8GB RAM (for AI model testing)
- Virtual environment management tool

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/eidolon.git
   cd eidolon
   ```

2. **Set up Virtual Environment**
   ```bash
   python3 -m venv eidolon_env
   source eidolon_env/bin/activate  # Windows: eidolon_env\Scripts\activate
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -r requirements-dev.txt
   pip install -e ".[dev]"
   ```

4. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

## 📝 Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes
- Follow the existing code style and patterns
- Add tests for new functionality
- Update documentation as needed
- Follow the SPARC methodology for complex features

### 3. Test Your Changes
```bash
# Run the test suite
pytest

# Run phase validation tests
python test_phase1.py
python test_phase2.py
python test_phase3.py

# Test specific functionality
python -m eidolon.cli.main status
```

### 4. Commit Your Changes
```bash
git add .
git commit -m "feat: add your feature description"
```

Follow conventional commit messages:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for test additions
- `refactor:` for code refactoring

### 5. Submit a Pull Request
1. Push to your fork
2. Create a pull request with:
   - Clear description of changes
   - Link to any related issues
   - Screenshots for UI changes
   - Test results

## 🧪 Testing Guidelines

### Test Structure
- Unit tests in `tests/`
- Integration tests in `test_phase*.py`
- Follow existing naming conventions

### Test Requirements
- All new features must have tests
- Maintain >90% test coverage
- Tests should be isolated and reproducible
- Include both positive and negative test cases

### Running Tests
```bash
# Full test suite
pytest

# With coverage
pytest --cov=src/eidolon --cov-report=html

# Specific test file
pytest tests/test_observer.py

# Phase validation
python test_phase1.py  # Foundation
python test_phase2.py  # Intelligent Capture
python test_phase3.py  # Local AI Integration
```

## 📋 Code Style Guidelines

### Python Style
- Follow PEP 8 guidelines
- Use Black for code formatting: `black src/`
- Use isort for import sorting: `isort src/`
- Add type hints to all functions
- Maximum line length: 88 characters

### Documentation Style
- Use Google-style docstrings
- Document all public APIs
- Include examples in docstrings
- Update README.md for user-facing changes

### Example Function Documentation
```python
def analyze_content(self, image_path: Union[str, Path], text: str = "") -> ContentAnalysis:
    """
    Analyze content using AI models and heuristics.
    
    Args:
        image_path: Path to the image file to analyze.
        text: Optional extracted text for enhanced analysis.
        
    Returns:
        ContentAnalysis: Analysis results with content type, description,
        confidence score, and optional vision analysis.
        
    Raises:
        ValueError: If image_path is invalid.
        RuntimeError: If analysis fails due to system issues.
        
    Example:
        >>> analyzer = Analyzer()
        >>> result = analyzer.analyze_content("screenshot.png")
        >>> print(f"Content type: {result.content_type}")
    """
```

## 🔒 Security Guidelines

### Data Handling
- Never log sensitive information
- Use environment variables for API keys
- Implement proper data sanitization
- Follow privacy-first principles

### AI Model Security
- Validate all inputs to AI models
- Implement proper error handling
- Use secure model loading practices
- Monitor resource usage

## 🚨 Issue Reporting

### Bug Reports
Include:
- System information (OS, Python version, RAM)
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs and error messages
- Screenshots if applicable

### Feature Requests
Include:
- Clear use case description
- Proposed implementation approach
- Potential impact on existing features
- Alternative solutions considered

## 📊 Development Phases

Eidolon development follows a structured phase approach:

- **Phase 1**: Foundation (✅ Complete)
- **Phase 2**: Intelligent Capture (✅ Complete)
- **Phase 3**: Local AI Integration (✅ Complete)
- **Phase 4**: Cloud AI & Semantic Memory (🚧 In Progress)
- **Phase 5**: Advanced Analytics (📋 Planned)
- **Phase 6**: MCP Integration & Agency (📋 Planned)
- **Phase 7**: Digital Twin (📋 Planned)

See [PROGRESS_PLAN.md](PROGRESS_PLAN.md) for detailed phase information.

## 💡 Architecture Notes

### Core Components
- **Observer**: Screenshot capture and monitoring
- **Analyzer**: OCR and AI-powered content analysis
- **Storage**: Database and file management
- **Interface**: CLI and future web/API interfaces

### AI Integration
- Local models (Florence-2, CLIP) for privacy
- Cloud API integration for advanced analysis
- Intelligent routing between local/cloud
- Resource-aware model loading

## 🤝 Community

- **Questions**: Use GitHub Discussions
- **Bug Reports**: Use GitHub Issues
- **Feature Requests**: Use GitHub Issues with feature label
- **Security Issues**: Email security@eidolon.ai

## 📄 License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.

---

Thank you for contributing to Eidolon! Your help makes this project better for everyone. 🚀