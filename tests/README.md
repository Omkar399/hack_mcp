# Eidolon Test Suite

## Test Organization

This directory contains all tests for the Eidolon AI Personal Assistant, organized into a clear hierarchy:

### Directory Structure

```
tests/
├── phase/           # Phase-based integration tests
│   ├── test_phase1.py   # Observer system tests
│   ├── test_phase2.py   # Analysis system tests  
│   ├── test_phase3.py   # Local AI integration tests
│   ├── test_phase4.py   # Cloud AI & semantic memory tests
│   └── test_phase5.py   # Advanced analytics tests
├── unit/            # Unit tests for individual components
│   ├── test_config.py   # Configuration system tests
│   └── test_observer.py # Observer component tests
├── integration/     # System integration tests
│   └── test_env_integration.py  # Environment setup tests
└── fixtures/        # Test data and fixtures
```

### Test Types

#### Phase Tests (`tests/phase/`)
Progressive integration tests that validate each development phase:
- **Phase 1**: Foundation and observer system
- **Phase 2**: Intelligent capture and OCR
- **Phase 3**: Local AI model integration
- **Phase 4**: Cloud AI APIs and vector database
- **Phase 5**: Advanced analytics and insights

#### Unit Tests (`tests/unit/`)
Isolated tests for individual components and modules:
- Configuration loading and validation
- Observer functionality
- Core component behavior
- Utility functions

#### Integration Tests (`tests/integration/`)
End-to-end tests that validate system integration:
- Environment setup and dependencies
- Cross-component interactions
- External service integrations
- Performance benchmarks

#### Fixtures (`tests/fixtures/`)
Shared test data, mock objects, and utilities:
- Sample screenshots and data
- Mock API responses
- Test configuration files
- Helper functions

## Running Tests

### All Tests
```bash
python -m pytest
```

### Specific Test Categories
```bash
# Phase tests only
python -m pytest tests/phase/

# Unit tests only  
python -m pytest tests/unit/

# Integration tests only
python -m pytest tests/integration/

# Specific phase
python -m pytest tests/phase/test_phase1.py
```

### With Coverage
```bash
python -m pytest --cov=eidolon --cov-report=html
```

## Test Configuration

Tests are configured via `pyproject.toml`:
- Test discovery patterns
- Coverage settings
- Async test support
- Reporting options

## Writing Tests

### Phase Tests
Follow the progressive development model:
- Each phase builds on the previous
- Test both isolated functionality and integration
- Include performance benchmarks
- Validate all phase requirements

### Unit Tests
Focus on single component behavior:
- Test all public methods
- Include edge cases and error conditions
- Mock external dependencies
- Aim for >90% coverage

### Integration Tests
Validate system-wide behavior:
- Test real component interactions
- Include setup/teardown procedures
- Test configuration scenarios
- Validate performance requirements

## Continuous Integration

All tests must pass before merging:
- Automated testing on push/PR
- Coverage requirements enforced
- Performance regression detection
- Cross-platform validation