# Eidolon Testing Guide

## Overview

This document provides comprehensive guidance for testing the Eidolon AI Personal Assistant, including the newly migrated EnrichMCP and chat functionalities.

## Test Structure

### Integration Tests

Located in `/tests/integration/`:

- **`test_mcp_integration.py`** - MCP server integration tests
- **`test_chat_integration.py`** - Chat functionality integration tests  
- **`test_e2e_workflow.py`** - End-to-end workflow tests
- **`test_env_integration.py`** - Environment integration tests

### Unit Tests

Located in `/tests/unit/`:

- **`test_mcp_server.py`** - MCP server unit tests
- **`test_enhanced_interface.py`** - Enhanced interface unit tests
- **`test_*.py`** - Existing component unit tests

### Phase Tests

Located in `/tests/phase/`:

- **`test_phase1.py`** through **`test_phase5.py`** - Phase-specific functionality tests

## Running Tests

### Basic Test Execution

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=eidolon --cov-report=html

# Run specific test file
python -m pytest tests/integration/test_mcp_integration.py

# Run specific test method
python -m pytest tests/integration/test_mcp_integration.py::TestMCPServerIntegration::test_mcp_screen_capture_endpoint
```

### Test Categories

```bash
# Run only integration tests
python -m pytest tests/integration/

# Run only unit tests
python -m pytest tests/unit/

# Run MCP-related tests
python -m pytest -m mcp

# Run chat-related tests
python -m pytest -m chat

# Run end-to-end tests
python -m pytest -m e2e

# Run async tests only
python -m pytest -k "async"
```

### Performance Testing

```bash
# Run tests with performance metrics
python -m pytest --benchmark-only

# Run load tests
python -m pytest tests/integration/test_e2e_workflow.py::TestCompleteWorkflow::test_performance_under_load
```

## Test Configuration

### Environment Variables

Set these environment variables for testing:

```bash
export EIDOLON_ENV=test
export GEMINI_API_KEY=test_key
export CLAUDE_API_KEY=test_key
export OPENAI_API_KEY=test_key
export OPENROUTER_API_KEY=test_key
```

### Test Configuration File

Create `tests/test_config.yaml`:

```yaml
test_environment:
  mock_external_apis: true
  use_temp_storage: true
  fast_mode: true
  
mcp_tests:
  server_port: 18080  # Different from production
  timeout: 5.0
  max_concurrent: 10
  
chat_tests:
  context_window: 1024  # Smaller for faster tests
  max_sessions: 5
  mock_providers: true
```

## Test Fixtures

### Core Fixtures (in `conftest.py`)

- **`mock_config`** - Complete configuration mock
- **`mock_memory_system`** - Memory system mock
- **`mock_cloud_api_manager`** - Cloud API manager mock
- **`sample_screenshot`** - Sample screenshot data
- **`temp_storage_dir`** - Temporary storage directories

### MCP Test Fixtures

- **`mock_mcp_server`** - MCP server mock
- **`sample_mcp_requests`** - Sample MCP requests
- **`sample_mcp_responses`** - Sample MCP responses
- **`mock_authenticator`** - Authentication mock
- **`mock_rate_limiter`** - Rate limiting mock

### Chat Test Fixtures

- **`mock_enhanced_interface`** - Enhanced interface mock
- **`mock_chat_session`** - Chat session mock
- **`sample_chat_messages`** - Sample chat messages
- **`mock_context_manager`** - Context manager mock

## Test Data

### Sample Data Generation

Use the provided test data generators:

```python
# Generate sample screenshots
screenshot = sample_screenshot_factory(
    width=1920,
    height=1080,
    content_type="code_editor"
)

# Generate sample chat conversations
conversation = sample_conversation_factory(
    message_count=10,
    topics=["python", "testing"],
    providers=["gemini", "claude"]
)

# Generate sample MCP requests
mcp_requests = sample_mcp_requests_factory(
    methods=["capture_screen", "search", "analyze"],
    count=5
)
```

### Mock Data

Test data is isolated and doesn't affect production:

- Screenshots stored in temporary directories
- Database operations use in-memory SQLite
- API calls are mocked
- No external network requests

## Testing Patterns

### Async Testing

```python
@pytest.mark.asyncio
async def test_async_functionality():
    """Test async functionality."""
    result = await async_function()
    assert result is not None
```

### Mocking External Services

```python
@patch('eidolon.models.cloud_api.genai')
async def test_with_mocked_api(mock_genai):
    """Test with mocked external API."""
    mock_genai.GenerativeModel.return_value.generate_content.return_value = "mock response"
    
    result = await function_that_uses_api()
    assert "mock response" in result
```

### Error Testing

```python
async def test_error_handling():
    """Test error handling."""
    with pytest.raises(SpecificError, match="expected error message"):
        await function_that_should_fail()
```

### Performance Testing

```python
async def test_performance():
    """Test performance requirements."""
    start_time = time.time()
    
    await function_to_test()
    
    duration = time.time() - start_time
    assert duration < 1.0  # Should complete in under 1 second
```

## Test Coverage

### Coverage Requirements

- **Unit Tests**: >90% coverage for new components
- **Integration Tests**: >80% coverage for workflows
- **End-to-End Tests**: Cover all major user journeys

### Coverage Reports

```bash
# Generate HTML coverage report
python -m pytest --cov=eidolon --cov-report=html

# View coverage report
open htmlcov/index.html

# Generate terminal coverage report
python -m pytest --cov=eidolon --cov-report=term-missing
```

### Coverage Exclusions

Some code is excluded from coverage requirements:

- Error handling for rare edge cases
- Platform-specific code paths
- Development/debug utilities
- External library integration stubs

## Debugging Tests

### Test Debugging

```bash
# Run with verbose output
python -m pytest -v

# Run with extra verbose output
python -m pytest -vv

# Run with print statements visible
python -m pytest -s

# Run with debugging on failure
python -m pytest --pdb

# Run specific test with debugging
python -m pytest --pdb tests/unit/test_mcp_server.py::TestMCPServer::test_server_initialization
```

### Log Output During Tests

```python
import logging

def test_with_logging(caplog):
    """Test with log capture."""
    with caplog.at_level(logging.INFO):
        function_that_logs()
    
    assert "expected log message" in caplog.text
```

## Continuous Integration

### GitHub Actions

```yaml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.9
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Run tests
        run: |
          python -m pytest --cov=eidolon --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v1
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

## Test Data Management

### Test Database

```python
@pytest.fixture
async def test_db():
    """Provide test database."""
    # Create temporary database
    db_path = ":memory:"  # In-memory SQLite
    
    # Initialize schema
    await init_database(db_path)
    
    yield db_path
    
    # Cleanup handled automatically for in-memory DB
```

### File System Testing

```python
def test_file_operations(tmp_path):
    """Test file operations with temporary directory."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")
    
    result = process_file(test_file)
    assert result is not None
```

## Performance Benchmarks

### Benchmark Tests

```python
def test_screenshot_capture_performance(benchmark):
    """Benchmark screenshot capture."""
    result = benchmark(capture_screenshot)
    assert result is not None

def test_search_performance(benchmark):
    """Benchmark search functionality."""
    result = benchmark(search_function, "test query")
    assert len(result) > 0
```

### Performance Targets

| Component | Target | Measurement |
|-----------|---------|------------|
| Screenshot Capture | <100ms | Response time |
| Local Analysis | <2s | Processing time |
| Search Query | <500ms | Response time |
| Chat Response | <5s | End-to-end time |
| MCP Request | <1s | Request handling |

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure PYTHONPATH includes project root
   export PYTHONPATH=$PWD:$PYTHONPATH
   ```

2. **Async Test Timeouts**
   ```python
   # Increase timeout for slow tests
   @pytest.mark.asyncio(timeout=30)
   async def test_slow_operation():
       pass
   ```

3. **Resource Cleanup**
   ```python
   # Always use proper cleanup
   @pytest.fixture
   async def resource():
       resource = create_resource()
       yield resource
       await resource.cleanup()
   ```

4. **Mock Issues**
   ```python
   # Reset mocks between tests
   def test_function(mock_obj):
       mock_obj.reset_mock()
       # Test code
   ```

### Test Environment Issues

- Ensure all required environment variables are set
- Check that test data directories are writable
- Verify mock external services are configured
- Confirm test database is accessible

## Contributing

### Adding New Tests

1. **Choose Appropriate Location**
   - Unit tests → `/tests/unit/`
   - Integration tests → `/tests/integration/`
   - End-to-end tests → `/tests/integration/`

2. **Follow Naming Conventions**
   - File: `test_component_name.py`
   - Class: `TestComponentName`
   - Method: `test_specific_functionality`

3. **Use Proper Fixtures**
   - Leverage existing fixtures when possible
   - Create new fixtures for reusable test data
   - Document fixture behavior

4. **Add Documentation**
   - Include docstrings for test classes/methods
   - Document complex test scenarios
   - Update this guide for new patterns

### Test Review Checklist

- [ ] Tests cover both success and failure cases
- [ ] Async tests use proper decorators
- [ ] Mocks are properly configured and reset
- [ ] Performance requirements are tested
- [ ] Error messages are descriptive
- [ ] Test data is isolated and clean
- [ ] Documentation is updated

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
- [pytest-cov](https://pytest-cov.readthedocs.io/)

---

*This testing guide is maintained alongside the Eidolon project. For questions or improvements, please see the project documentation or contribute via pull request.*