"""
Global test fixtures for Eidolon test suite.

This module provides common fixtures used across multiple test files
to ensure consistent test setup and reduce code duplication.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any

# Import classes that need to be mocked
from eidolon.models.cloud_api import CloudAPIResponse


# Configuration fixtures
@pytest.fixture
def mock_config():
    """Mock configuration object used across tests."""
    config = Mock()
    
    # Observer configuration
    config.observer = Mock()
    config.observer.capture_interval = 10
    config.observer.activity_threshold = 0.05
    config.observer.storage_path = "./data/screenshots"
    config.observer.max_storage_gb = 50.0
    config.observer.monitor_keyboard = True
    config.observer.monitor_mouse = True
    config.observer.sensitive_patterns = ["password", "api_key", "ssn"]
    
    # Analysis configuration
    config.analysis = Mock()
    config.analysis.local_models = {
        "vision": "microsoft/florence-2-base",
        "clip": "openai/clip-vit-base-patch32",
        "embedding": "sentence-transformers/all-MiniLM-L6-v2"
    }
    config.analysis.cloud_apis = {
        "gemini_key": "test_gemini_key",
        "claude_key": "test_claude_key", 
        "openai_key": "test_openai_key",
        "openrouter_key": "test_openrouter_key"
    }
    config.analysis.preferred_providers = ["gemini", "claude", "openai"]
    config.analysis.routing = {
        "importance_threshold": 0.7,
        "cost_limit_daily": 10.0,
        "local_first": True
    }
    config.analysis.ocr = {
        "engine": "tesseract",
        "confidence_threshold": 0.8
    }
    
    # Memory configuration
    config.memory = Mock()
    config.memory.vector_db = "chromadb"
    config.memory.chunk_size = 512
    config.memory.overlap = 50
    config.memory.metadata_db = "sqlite"
    config.memory.db_path = "./data/eidolon.db"
    config.memory.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    config.memory.search = {
        "max_results": 50,
        "enable_semantic_search": True
    }
    
    # Privacy configuration
    config.privacy = Mock()
    config.privacy.local_only_mode = False
    config.privacy.auto_redaction = True
    config.privacy.data_retention_days = 365
    config.privacy.encrypt_at_rest = True
    config.privacy.pause_on_sensitive_apps = True
    config.privacy.sensitive_patterns = ["password", "api_key", "ssn"]
    config.privacy.excluded_apps = ["1Password", "Keychain Access"]
    
    # Logging configuration
    config.logging = Mock()
    config.logging.level = "INFO"
    config.logging.file_path = "./logs/eidolon.log"
    config.logging.max_size_mb = 100
    config.logging.backup_count = 5
    
    # Interface configuration
    config.interface = Mock()
    config.interface.api_port = 8000
    config.interface.web_port = 3000
    config.interface.enable_api = True
    config.interface.enable_web = True
    
    # Chat configuration (for new features)
    config.interface.chat = Mock()
    config.interface.chat.context_window = 4096
    config.interface.chat.max_tokens = 2048
    config.interface.chat.default_provider = "gemini"
    config.interface.chat.enable_streaming = True
    config.interface.chat.persistence_path = "./data/chat_sessions"
    
    # MCP configuration (for new features)
    config.mcp = Mock()
    config.mcp.server_port = 8080
    config.mcp.server_host = "localhost"
    config.mcp.enable_authentication = True
    config.mcp.rate_limit_requests_per_minute = 60
    config.mcp.max_concurrent_requests = 50
    
    return config


# Cloud API fixtures
@pytest.fixture
def mock_genai():
    """Mock the Google Generative AI module."""
    with patch('eidolon.models.cloud_api.genai') as mock:
        mock_model = Mock()
        mock.GenerativeModel.return_value = mock_model
        
        # Mock Image class for Gemini
        mock_image = Mock()
        mock.Image = Mock(return_value=mock_image)
        
        yield mock, mock_model


@pytest.fixture
def mock_anthropic():
    """Mock the Anthropic client."""
    with patch('eidolon.models.cloud_api.Anthropic') as mock:
        mock_client = Mock()
        mock.return_value = mock_client
        yield mock, mock_client


@pytest.fixture
def mock_openai():
    """Mock the OpenAI client."""
    with patch('eidolon.models.cloud_api.OpenAI') as mock:
        mock_client = Mock()
        mock.return_value = mock_client
        yield mock, mock_client


@pytest.fixture
def mock_aiohttp():
    """Mock aiohttp for HTTP requests."""
    with patch('eidolon.models.cloud_api.aiohttp.ClientSession') as mock:
        mock_session = AsyncMock()
        mock.return_value.__aenter__.return_value = mock_session
        yield mock_session


# File system fixtures
@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for configuration files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_image_file(tmp_path):
    """Create a mock image file."""
    image_path = tmp_path / "test_image.png"
    image_path.write_bytes(b"fake_image_data")
    return str(image_path)


@pytest.fixture
def sample_screenshot():
    """Create a sample screenshot object for testing."""
    screenshot = Mock()
    screenshot.timestamp = "2024-01-01T12:00:00"
    screenshot.filename = "test_screenshot.png" 
    screenshot.width = 1920
    screenshot.height = 1080
    screenshot.file_path = "/tmp/test_screenshot.png"
    screenshot.size_bytes = 1024 * 1024  # 1MB
    return screenshot


# Database fixtures
@pytest.fixture
def mock_database():
    """Mock database connection for testing."""
    db = Mock()
    db.execute = Mock()
    db.fetchone = Mock()
    db.fetchall = Mock()
    db.commit = Mock()
    db.close = Mock()
    return db


@pytest.fixture
def mock_vector_db():
    """Mock vector database for testing."""
    vector_db = Mock()
    vector_db.add = Mock()
    vector_db.search = Mock()
    vector_db.update = Mock()
    vector_db.delete = Mock()
    vector_db.get_collection = Mock()
    return vector_db


# Response fixtures
@pytest.fixture
def sample_cloud_response():
    """Create a sample CloudAPIResponse for testing."""
    return CloudAPIResponse(
        content="This is a test response",
        model="test-model",
        provider="test-provider",
        confidence=0.9,
        usage={"tokens": 100, "cost": 0.01},
        metadata={"temperature": 0.7}
    )


# Environment fixtures
@pytest.fixture
def clean_environment():
    """Provide clean environment variables for testing."""
    original_env = os.environ.copy()
    
    # Clear relevant environment variables
    test_env_vars = [
        "GEMINI_API_KEY", "CLAUDE_API_KEY", "OPENAI_API_KEY",
        "OPENROUTER_API_KEY", "EIDOLON_CONFIG_PATH"
    ]
    
    for var in test_env_vars:
        if var in os.environ:
            del os.environ[var]
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def test_env_vars():
    """Set up test environment variables."""
    test_vars = {
        "GEMINI_API_KEY": "test_gemini_key",
        "CLAUDE_API_KEY": "test_claude_key", 
        "OPENAI_API_KEY": "test_openai_key",
        "OPENROUTER_API_KEY": "test_openrouter_key"
    }
    
    original_env = {}
    for key, value in test_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield test_vars
    
    # Restore original values
    for key, original_value in original_env.items():
        if original_value is None:
            if key in os.environ:
                del os.environ[key]
        else:
            os.environ[key] = original_value


# Local model fixtures  
@pytest.fixture
def mock_local_models():
    """Mock local AI models for testing."""
    models = Mock()
    models.florence = Mock()
    models.clip = Mock()
    models.embedding = Mock()
    
    # Mock model methods
    models.florence.analyze_image = AsyncMock()
    models.clip.encode_image = Mock()
    models.clip.encode_text = Mock()
    models.embedding.encode = Mock()
    
    return models


# Observer fixtures
@pytest.fixture
def mock_observer():
    """Mock Observer for testing."""
    observer = Mock()
    observer.start_monitoring = Mock()
    observer.stop_monitoring = Mock()
    observer.capture_screenshot = Mock()
    observer.is_running = False
    observer.last_capture_time = None
    return observer


# Memory fixtures
@pytest.fixture
def mock_memory_system():
    """Mock MemorySystem for testing."""
    memory = Mock()
    memory.store = AsyncMock()
    memory.search = AsyncMock()
    memory.retrieve = AsyncMock()
    memory.update = AsyncMock()
    memory.delete = AsyncMock()
    return memory


# Analysis fixtures
@pytest.fixture
def mock_analyzer():
    """Mock Analyzer for testing."""
    analyzer = Mock()
    analyzer.analyze_screenshot = AsyncMock()
    analyzer.extract_text = AsyncMock()
    analyzer.classify_content = AsyncMock()
    analyzer.detect_changes = Mock()
    return analyzer


# Async fixtures
@pytest.fixture
def mock_async_client():
    """Mock async HTTP client for testing."""
    client = AsyncMock()
    client.get = AsyncMock()
    client.post = AsyncMock()
    client.put = AsyncMock()
    client.delete = AsyncMock()
    client.close = AsyncMock()
    return client


# Performance monitoring fixtures
@pytest.fixture
def mock_performance_monitor():
    """Mock performance monitoring for tests."""
    monitor = Mock()
    monitor.start_timing = Mock()
    monitor.end_timing = Mock()
    monitor.record_metric = Mock()
    monitor.get_metrics = Mock(return_value={})
    return monitor


# Logging fixtures
@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    logger = Mock()
    logger.debug = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.critical = Mock()
    return logger


# Temporary file fixtures
@pytest.fixture
def temp_storage_dir(tmp_path):
    """Create temporary storage directories for testing."""
    storage_dirs = {
        'screenshots': tmp_path / "screenshots",
        'data': tmp_path / "data", 
        'logs': tmp_path / "logs",
        'models': tmp_path / "models"
    }
    
    for dir_path in storage_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return storage_dirs


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "cloud_api: mark test as requiring cloud API"
    )
    config.addinivalue_line(
        "markers", "local_model: mark test as requiring local AI models"
    )


# Fixture to disable external network calls
@pytest.fixture(autouse=True)
def disable_network_calls():
    """Automatically disable external network calls in tests."""
    with patch('requests.get'), \
         patch('requests.post'), \
         patch('requests.put'), \
         patch('requests.delete'), \
         patch('aiohttp.ClientSession'):
        yield


# Coverage configuration
pytest_plugins = ["pytest_cov"]


# New fixtures for MCP and Chat testing
@pytest.fixture
def mock_cloud_api_manager():
    """Mock CloudAPIManager for chat testing."""
    manager = AsyncMock()
    manager.analyze_with_context = AsyncMock()
    manager.stream_with_context = AsyncMock()
    manager.get_available_providers = Mock(return_value=["gemini", "claude", "openai"])
    manager.check_api_availability = AsyncMock(return_value=True)
    return manager


@pytest.fixture
def mock_mcp_server():
    """Mock MCP server for testing."""
    server = Mock()
    server.is_running = False
    server.port = None
    server.host = None
    server.start = AsyncMock()
    server.shutdown = AsyncMock()
    server.handle_request = AsyncMock()
    server.get_metrics = AsyncMock(return_value={
        "total_requests": 0,
        "requests_by_method": {},
        "error_rate": 0.0,
        "average_response_time": 0.0
    })
    server.configure_rate_limit = Mock()
    server.verify_auth = Mock(return_value=True)
    return server


@pytest.fixture
def mock_chat_session():
    """Mock chat session for testing."""
    session = Mock()
    session.id = "test-session-123"
    session.created_at = "2024-01-01T12:00:00"
    session.context_window = 4096
    session.max_tokens = 2048
    session.enable_memory = True
    session.messages = []
    session.metadata = {}
    session.add_message = Mock()
    session.get_active_context = Mock(return_value={
        "messages": [],
        "token_count": 0
    })
    return session


@pytest.fixture
def mock_enhanced_interface():
    """Mock enhanced interface for testing."""
    interface = AsyncMock()
    interface.create_chat_session = AsyncMock()
    interface.send_message = AsyncMock()
    interface.send_message_stream = AsyncMock()
    interface.get_chat_session = AsyncMock()
    interface.get_conversation_history = AsyncMock()
    interface.export_conversation = AsyncMock()
    interface.import_conversation = AsyncMock()
    interface.save_sessions = AsyncMock()
    interface.load_sessions = AsyncMock()
    interface.configure_rate_limit = Mock()
    interface.configure_persistence = Mock()
    return interface


@pytest.fixture
def sample_chat_messages():
    """Sample chat messages for testing."""
    return [
        {
            "id": "msg-1",
            "role": "user",
            "content": "Hello, how are you?",
            "timestamp": "2024-01-01T12:00:00",
            "provider": None,
            "confidence": None
        },
        {
            "id": "msg-2", 
            "role": "assistant",
            "content": "Hello! I'm doing well, thank you for asking. How can I help you today?",
            "timestamp": "2024-01-01T12:00:05",
            "provider": "gemini",
            "confidence": 0.92
        },
        {
            "id": "msg-3",
            "role": "user", 
            "content": "Can you help me with Python programming?",
            "timestamp": "2024-01-01T12:01:00",
            "provider": None,
            "confidence": None
        },
        {
            "id": "msg-4",
            "role": "assistant",
            "content": "Absolutely! I'd be happy to help you with Python programming. What specific topic or problem would you like assistance with?",
            "timestamp": "2024-01-01T12:01:03",
            "provider": "gemini",
            "confidence": 0.95
        }
    ]


@pytest.fixture
def sample_mcp_requests():
    """Sample MCP requests for testing."""
    return [
        {
            "method": "capture_screen",
            "params": {"area": "full", "format": "png"},
            "id": "capture-123",
            "auth": {"token": "test-token"}
        },
        {
            "method": "search",
            "params": {"query": "python testing", "limit": 10},
            "id": "search-456",
            "auth": {"token": "test-token"}
        },
        {
            "method": "analyze_context",
            "params": {"window_minutes": 30, "include_screenshots": True},
            "id": "analyze-789",
            "auth": {"token": "test-token"}
        }
    ]


@pytest.fixture
def sample_mcp_responses():
    """Sample MCP responses for testing."""
    return [
        {
            "success": True,
            "result": {
                "screenshot_id": "screen-123",
                "timestamp": "2024-01-01T12:00:00",
                "dimensions": {"width": 1920, "height": 1080}
            },
            "request_id": "capture-123"
        },
        {
            "success": True,
            "result": {
                "results": [
                    {"id": "result-1", "content": "Python test content", "confidence": 0.9},
                    {"id": "result-2", "content": "More Python content", "confidence": 0.85}
                ],
                "query": "python testing",
                "total_results": 2
            },
            "request_id": "search-456"
        },
        {
            "success": False,
            "error": {
                "code": "INVALID_PARAMS",
                "message": "Missing required parameter: query",
                "details": {}
            },
            "request_id": "error-request"
        }
    ]


@pytest.fixture
def mock_context_manager():
    """Mock context manager for testing."""
    manager = AsyncMock()
    manager.get_context = AsyncMock(return_value={
        "recent_screenshots": [
            {"id": "screen-1", "content": "VS Code with Python code"},
            {"id": "screen-2", "content": "Terminal with test output"}
        ],
        "extracted_text": "import pytest\ndef test_example():\n    assert True",
        "applications": ["VS Code", "Terminal", "Chrome"],
        "activities": ["coding", "testing", "browsing"]
    })
    manager.get_relevant_context = AsyncMock()
    manager.summarize_context = AsyncMock()
    return manager


@pytest.fixture
def mock_rate_limiter():
    """Mock rate limiter for testing."""
    limiter = Mock()
    limiter.check_rate_limit = AsyncMock(return_value=Mock(
        allowed=True,
        remaining=10,
        retry_after=0
    ))
    limiter.configure = Mock()
    return limiter


@pytest.fixture
def mock_authenticator():
    """Mock authenticator for testing."""
    auth = AsyncMock()
    auth.authenticate = AsyncMock(return_value=Mock(
        success=True,
        user="test_user",
        permissions=["read", "write"],
        error=None
    ))
    auth.has_permission = Mock(return_value=True)
    return auth


@pytest.fixture
def async_test_timeout():
    """Configure timeout for async tests."""
    return 10.0  # 10 seconds timeout for async tests


# Test markers are configured in pytest.ini or pyproject.toml
# Markers: mcp, chat, e2e, integration, slow, cloud_api, local_model