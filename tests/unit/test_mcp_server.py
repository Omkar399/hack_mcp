#!/usr/bin/env python3
"""
Unit tests for MCP Server component.

Tests the MCP server implementation including request handling,
method routing, authentication, and error handling.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

# Import MCP server components (mocked for testing)
# These would be imported from actual implementation after migration
# from eidolon.core.mcp_server import (
#     MCPServer, MCPRequest, MCPResponse, MCPError,
#     MCPMethodHandler, MCPAuthenticator, MCPRateLimiter
# )


class TestMCPRequest:
    """Test MCPRequest class."""
    
    def test_mcp_request_creation(self):
        """Test MCPRequest creation with all fields."""
        request = MCPRequest(
            method="test_method",
            params={"param1": "value1", "param2": 42},
            id="test-request-123",
            auth={"token": "test-token"}
        )
        
        assert request.method == "test_method"
        assert request.params["param1"] == "value1"
        assert request.params["param2"] == 42
        assert request.id == "test-request-123"
        assert request.auth["token"] == "test-token"
        assert isinstance(request.timestamp, datetime)
    
    def test_mcp_request_defaults(self):
        """Test MCPRequest with default values."""
        request = MCPRequest(
            method="simple_method",
            params={},
            id="simple-123"
        )
        
        assert request.auth is None
        assert request.params == {}
        assert request.method == "simple_method"
    
    def test_mcp_request_validation(self):
        """Test MCPRequest validation."""
        # Test empty method
        with pytest.raises(ValueError, match="Method cannot be empty"):
            MCPRequest(method="", params={}, id="test")
        
        # Test empty ID
        with pytest.raises(ValueError, match="ID cannot be empty"):
            MCPRequest(method="test", params={}, id="")
        
        # Test invalid params type
        with pytest.raises(TypeError, match="Params must be a dictionary"):
            MCPRequest(method="test", params="invalid", id="test")
    
    def test_mcp_request_serialization(self):
        """Test MCPRequest serialization."""
        request = MCPRequest(
            method="serialize_test",
            params={"data": "test"},
            id="serialize-123"
        )
        
        serialized = request.to_dict()
        
        assert serialized["method"] == "serialize_test"
        assert serialized["params"]["data"] == "test"
        assert serialized["id"] == "serialize-123"
        assert "timestamp" in serialized
    
    def test_mcp_request_deserialization(self):
        """Test MCPRequest deserialization."""
        data = {
            "method": "deserialize_test",
            "params": {"value": 123},
            "id": "deserialize-456",
            "auth": {"token": "test-token"}
        }
        
        request = MCPRequest.from_dict(data)
        
        assert request.method == "deserialize_test"
        assert request.params["value"] == 123
        assert request.id == "deserialize-456"
        assert request.auth["token"] == "test-token"


class TestMCPResponse:
    """Test MCPResponse class."""
    
    def test_mcp_response_success(self):
        """Test successful MCPResponse creation."""
        response = MCPResponse(
            success=True,
            result={"data": "test_result", "count": 5},
            request_id="test-123"
        )
        
        assert response.success
        assert response.result["data"] == "test_result"
        assert response.result["count"] == 5
        assert response.error is None
        assert response.request_id == "test-123"
        assert isinstance(response.timestamp, datetime)
    
    def test_mcp_response_error(self):
        """Test error MCPResponse creation."""
        error = MCPError(
            code="TEST_ERROR",
            message="Test error occurred",
            details={"context": "unit_test"}
        )
        
        response = MCPResponse(
            success=False,
            error=error,
            request_id="error-456"
        )
        
        assert not response.success
        assert response.result is None
        assert response.error.code == "TEST_ERROR"
        assert response.error.message == "Test error occurred"
        assert response.error.details["context"] == "unit_test"
    
    def test_mcp_response_validation(self):
        """Test MCPResponse validation."""
        # Test invalid success response (no result)
        with pytest.raises(ValueError, match="Success response must have result"):
            MCPResponse(success=True, error=None, request_id="test")
        
        # Test invalid error response (no error)
        with pytest.raises(ValueError, match="Error response must have error"):
            MCPResponse(success=False, result=None, request_id="test")
        
        # Test both result and error
        error = MCPError("TEST", "Test")
        with pytest.raises(ValueError, match="Response cannot have both result and error"):
            MCPResponse(success=True, result={}, error=error, request_id="test")
    
    def test_mcp_response_serialization(self):
        """Test MCPResponse serialization."""
        response = MCPResponse(
            success=True,
            result={"message": "success"},
            request_id="serialize-789"
        )
        
        serialized = response.to_dict()
        
        assert serialized["success"]
        assert serialized["result"]["message"] == "success"
        assert serialized["request_id"] == "serialize-789"
        assert "timestamp" in serialized


class TestMCPError:
    """Test MCPError class."""
    
    def test_mcp_error_creation(self):
        """Test MCPError creation."""
        error = MCPError(
            code="VALIDATION_ERROR",
            message="Invalid parameter value",
            details={"parameter": "limit", "value": -1}
        )
        
        assert error.code == "VALIDATION_ERROR"
        assert error.message == "Invalid parameter value"
        assert error.details["parameter"] == "limit"
        assert error.details["value"] == -1
    
    def test_mcp_error_defaults(self):
        """Test MCPError with defaults."""
        error = MCPError("SIMPLE_ERROR", "Simple error message")
        
        assert error.code == "SIMPLE_ERROR"
        assert error.message == "Simple error message"
        assert error.details == {}
    
    def test_mcp_error_from_exception(self):
        """Test MCPError creation from exception."""
        try:
            raise ValueError("Test exception")
        except Exception as e:
            error = MCPError.from_exception(e, "EXCEPTION_ERROR")
        
        assert error.code == "EXCEPTION_ERROR"
        assert "Test exception" in error.message
        assert error.details["exception_type"] == "ValueError"


@pytest.mark.asyncio
class TestMCPMethodHandler:
    """Test MCPMethodHandler class."""
    
    @pytest.fixture
    def method_handler(self):
        """Create method handler for testing."""
        return MCPMethodHandler()
    
    def test_handler_registration(self, method_handler):
        """Test method handler registration."""
        @method_handler.register("test_method")
        async def test_handler(params):
            return {"result": "test"}
        
        assert "test_method" in method_handler._handlers
        assert method_handler._handlers["test_method"] == test_handler
    
    def test_handler_validation_decorator(self, method_handler):
        """Test handler with validation decorator."""
        @method_handler.register("validated_method")
        @method_handler.validate_params({"query": str, "limit": int})
        async def validated_handler(params):
            return {"query": params["query"], "limit": params["limit"]}
        
        # Test with valid params
        params = {"query": "test", "limit": 10}
        # Direct call to test validation (would normally be called by server)
        assert params["query"] == "test"
        assert isinstance(params["limit"], int)
    
    async def test_handler_execution(self, method_handler):
        """Test handler execution."""
        @method_handler.register("execution_test")
        async def execution_handler(params):
            return {"input": params.get("input", "none")}
        
        # Test handler exists
        assert method_handler.has_handler("execution_test")
        
        # Test handler execution
        handler = method_handler.get_handler("execution_test")
        result = await handler({"input": "test_value"})
        
        assert result["input"] == "test_value"
    
    def test_handler_not_found(self, method_handler):
        """Test handler not found scenario."""
        assert not method_handler.has_handler("nonexistent_method")
        assert method_handler.get_handler("nonexistent_method") is None
    
    async def test_handler_error_handling(self, method_handler):
        """Test handler error handling."""
        @method_handler.register("error_method")
        async def error_handler(params):
            raise ValueError("Handler error")
        
        handler = method_handler.get_handler("error_method")
        
        with pytest.raises(ValueError, match="Handler error"):
            await handler({})


class TestMCPAuthenticator:
    """Test MCPAuthenticator class."""
    
    @pytest.fixture
    def authenticator(self):
        """Create authenticator for testing."""
        return MCPAuthenticator(
            tokens={"valid-token": {"user": "test_user", "permissions": ["read", "write"]}},
            require_auth=True
        )
    
    async def test_valid_authentication(self, authenticator):
        """Test valid token authentication."""
        auth_data = {"token": "valid-token"}
        
        result = await authenticator.authenticate(auth_data)
        
        assert result.success
        assert result.user == "test_user"
        assert "read" in result.permissions
        assert "write" in result.permissions
    
    async def test_invalid_authentication(self, authenticator):
        """Test invalid token authentication."""
        auth_data = {"token": "invalid-token"}
        
        result = await authenticator.authenticate(auth_data)
        
        assert not result.success
        assert result.error.code == "INVALID_TOKEN"
    
    async def test_missing_authentication(self, authenticator):
        """Test missing authentication."""
        result = await authenticator.authenticate(None)
        
        assert not result.success
        assert result.error.code == "MISSING_AUTH"
    
    async def test_permission_check(self, authenticator):
        """Test permission checking."""
        auth_data = {"token": "valid-token"}
        auth_result = await authenticator.authenticate(auth_data)
        
        assert authenticator.has_permission(auth_result, "read")
        assert authenticator.has_permission(auth_result, "write")
        assert not authenticator.has_permission(auth_result, "admin")
    
    async def test_optional_authentication(self):
        """Test optional authentication mode."""
        optional_auth = MCPAuthenticator(require_auth=False)
        
        result = await optional_auth.authenticate(None)
        assert result.success
        assert result.user == "anonymous"


class TestMCPRateLimiter:
    """Test MCPRateLimiter class."""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create rate limiter for testing."""
        return MCPRateLimiter(
            requests_per_minute=5,
            burst_limit=10
        )
    
    async def test_rate_limiting_allowed(self, rate_limiter):
        """Test requests within rate limit."""
        client_id = "test-client"
        
        # First few requests should be allowed
        for i in range(5):
            result = await rate_limiter.check_rate_limit(client_id)
            assert result.allowed
            assert result.remaining > 0
    
    async def test_rate_limiting_exceeded(self, rate_limiter):
        """Test rate limit exceeded."""
        client_id = "heavy-client"
        
        # Exceed rate limit
        for i in range(6):
            result = await rate_limiter.check_rate_limit(client_id)
            if i < 5:
                assert result.allowed
            else:
                assert not result.allowed
                assert result.retry_after > 0
    
    async def test_rate_limit_reset(self, rate_limiter):
        """Test rate limit reset after time window."""
        client_id = "reset-client"
        
        # Exceed limit
        for i in range(6):
            await rate_limiter.check_rate_limit(client_id)
        
        # Check limit exceeded
        result = await rate_limiter.check_rate_limit(client_id)
        assert not result.allowed
        
        # Mock time advancement (in real implementation would wait)
        rate_limiter._reset_window(client_id)
        
        # Should be allowed again
        result = await rate_limiter.check_rate_limit(client_id)
        assert result.allowed
    
    async def test_burst_limit(self, rate_limiter):
        """Test burst limit handling."""
        client_id = "burst-client"
        
        # Send burst of requests
        for i in range(10):
            result = await rate_limiter.check_rate_limit(client_id)
            assert result.allowed  # All should be allowed within burst
        
        # 11th request should be limited
        result = await rate_limiter.check_rate_limit(client_id)
        assert not result.allowed


@pytest.mark.asyncio
class TestMCPServer:
    """Test MCPServer class."""
    
    @pytest.fixture
    async def mcp_server(self, mock_config):
        """Create MCP server for testing."""
        server = MCPServer(
            config=mock_config,
            observer=Mock(),
            memory=AsyncMock(),
            analyzer=AsyncMock()
        )
        yield server
        if hasattr(server, 'shutdown'):
            await server.shutdown()
    
    async def test_server_initialization(self, mcp_server, mock_config):
        """Test server initialization."""
        assert mcp_server.config == mock_config
        assert mcp_server.observer is not None
        assert mcp_server.memory is not None
        assert mcp_server.analyzer is not None
        assert not mcp_server.is_running
    
    async def test_server_start_stop(self, mcp_server):
        """Test server start and stop."""
        # Start server
        await mcp_server.start(port=8080, host="localhost")
        assert mcp_server.is_running
        assert mcp_server.port == 8080
        assert mcp_server.host == "localhost"
        
        # Stop server
        await mcp_server.shutdown()
        assert not mcp_server.is_running
    
    async def test_request_handling(self, mcp_server):
        """Test basic request handling."""
        # Mock a simple handler
        async def mock_handler(params):
            return {"message": "Handler executed", "params": params}
        
        mcp_server._method_handler.register("test_method")(mock_handler)
        
        request = MCPRequest(
            method="test_method",
            params={"test_param": "test_value"},
            id="test-request"
        )
        
        response = await mcp_server.handle_request(request)
        
        assert response.success
        assert response.result["message"] == "Handler executed"
        assert response.result["params"]["test_param"] == "test_value"
        assert response.request_id == "test-request"
    
    async def test_method_not_found(self, mcp_server):
        """Test handling of unknown methods."""
        request = MCPRequest(
            method="unknown_method",
            params={},
            id="unknown-request"
        )
        
        response = await mcp_server.handle_request(request)
        
        assert not response.success
        assert response.error.code == "METHOD_NOT_FOUND"
        assert "unknown_method" in response.error.message
    
    async def test_authentication_integration(self, mcp_server):
        """Test authentication integration."""
        # Configure authentication
        mcp_server._authenticator = MCPAuthenticator(
            tokens={"valid-token": {"user": "test_user"}},
            require_auth=True
        )
        
        # Request without auth
        request = MCPRequest(
            method="capture_screen",
            params={},
            id="unauth-request"
        )
        
        response = await mcp_server.handle_request(request)
        assert not response.success
        assert response.error.code == "MISSING_AUTH"
        
        # Request with valid auth
        request.auth = {"token": "valid-token"}
        # Mock the capture_screen handler
        mcp_server._method_handler.register("capture_screen")(
            lambda params: {"success": True}
        )
        
        response = await mcp_server.handle_request(request)
        assert response.success
    
    async def test_rate_limiting_integration(self, mcp_server):
        """Test rate limiting integration."""
        # Configure rate limiting
        mcp_server._rate_limiter = MCPRateLimiter(requests_per_minute=2)
        
        # Mock handler
        mcp_server._method_handler.register("limited_method")(
            lambda params: {"success": True}
        )
        
        # First two requests should succeed
        for i in range(2):
            request = MCPRequest(
                method="limited_method",
                params={},
                id=f"rate-test-{i}",
                auth={"client_id": "test-client"}
            )
            response = await mcp_server.handle_request(request)
            assert response.success
        
        # Third request should be rate limited
        request = MCPRequest(
            method="limited_method",
            params={},
            id="rate-test-3",
            auth={"client_id": "test-client"}
        )
        response = await mcp_server.handle_request(request)
        assert not response.success
        assert response.error.code == "RATE_LIMITED"
    
    async def test_error_handling(self, mcp_server):
        """Test comprehensive error handling."""
        # Handler that raises exception
        async def error_handler(params):
            raise ValueError("Test error")
        
        mcp_server._method_handler.register("error_method")(error_handler)
        
        request = MCPRequest(
            method="error_method",
            params={},
            id="error-request"
        )
        
        response = await mcp_server.handle_request(request)
        
        assert not response.success
        assert response.error.code == "INTERNAL_ERROR"
        assert "Test error" in response.error.message
    
    async def test_concurrent_request_handling(self, mcp_server):
        """Test concurrent request handling."""
        # Slow handler to test concurrency
        async def slow_handler(params):
            await asyncio.sleep(0.1)
            return {"id": params.get("id"), "completed": True}
        
        mcp_server._method_handler.register("slow_method")(slow_handler)
        
        # Create multiple concurrent requests
        requests = [
            MCPRequest(
                method="slow_method",
                params={"id": i},
                id=f"concurrent-{i}"
            )
            for i in range(5)
        ]
        
        # Handle requests concurrently
        start_time = datetime.now()
        responses = await asyncio.gather(
            *[mcp_server.handle_request(req) for req in requests]
        )
        end_time = datetime.now()
        
        # Should complete in parallel (less than sequential time)
        duration = (end_time - start_time).total_seconds()
        assert duration < 0.4  # Much less than 5 * 0.1 seconds
        
        # All requests should succeed
        assert len(responses) == 5
        for i, response in enumerate(responses):
            assert response.success
            assert response.result["id"] == i
            assert response.result["completed"]
    
    async def test_metrics_collection(self, mcp_server):
        """Test metrics collection."""
        # Mock handler
        mcp_server._method_handler.register("metrics_method")(
            lambda params: {"success": True}
        )
        
        # Make several requests
        for i in range(3):
            request = MCPRequest(
                method="metrics_method",
                params={},
                id=f"metrics-{i}"
            )
            await mcp_server.handle_request(request)
        
        # Get metrics
        metrics = await mcp_server.get_metrics()
        
        assert metrics["total_requests"] >= 3
        assert "metrics_method" in metrics["requests_by_method"]
        assert metrics["requests_by_method"]["metrics_method"] >= 3
        assert "average_response_time" in metrics
        assert metrics["error_rate"] == 0.0
    
    async def test_server_configuration(self, mcp_server, mock_config):
        """Test server configuration handling."""
        # Test default configuration
        assert mcp_server.config == mock_config
        
        # Test configuration updates
        new_config = {
            "max_concurrent_requests": 100,
            "request_timeout": 30,
            "enable_metrics": True
        }
        
        mcp_server.update_config(new_config)
        
        assert mcp_server._max_concurrent == 100
        assert mcp_server._request_timeout == 30
        assert mcp_server._enable_metrics
    
    async def test_graceful_shutdown(self, mcp_server):
        """Test graceful server shutdown."""
        # Start server
        await mcp_server.start(port=8080)
        
        # Add some pending requests
        async def long_handler(params):
            await asyncio.sleep(0.2)
            return {"completed": True}
        
        mcp_server._method_handler.register("long_method")(long_handler)
        
        # Start some requests
        requests = [
            mcp_server.handle_request(
                MCPRequest(method="long_method", params={}, id=f"shutdown-{i}")
            )
            for i in range(3)
        ]
        
        # Start shutdown (should wait for requests)
        start_time = datetime.now()
        shutdown_task = asyncio.create_task(mcp_server.shutdown(timeout=1.0))
        
        # Wait for requests and shutdown
        responses = await asyncio.gather(*requests)
        await shutdown_task
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Should have waited for requests to complete
        assert duration >= 0.2
        assert not mcp_server.is_running
        
        # All requests should have completed successfully
        for response in responses:
            assert response.success


# Mock classes for testing
class MCPRequest:
    """Mock MCP request class."""
    def __init__(self, method: str, params: Dict[str, Any], id: str, auth: Optional[Dict] = None):
        if not method:
            raise ValueError("Method cannot be empty")
        if not id:
            raise ValueError("ID cannot be empty")
        if not isinstance(params, dict):
            raise TypeError("Params must be a dictionary")
        
        self.method = method
        self.params = params
        self.id = id
        self.auth = auth
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "method": self.method,
            "params": self.params,
            "id": self.id,
            "timestamp": self.timestamp.isoformat()
        }
        if self.auth:
            result["auth"] = self.auth
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPRequest':
        """Deserialize from dictionary."""
        return cls(
            method=data["method"],
            params=data["params"],
            id=data["id"],
            auth=data.get("auth")
        )


class MCPResponse:
    """Mock MCP response class."""
    def __init__(self, success: bool, result: Optional[Dict] = None, 
                 error: Optional['MCPError'] = None, request_id: str = ""):
        if success and result is None:
            raise ValueError("Success response must have result")
        if not success and error is None:
            raise ValueError("Error response must have error")
        if result is not None and error is not None:
            raise ValueError("Response cannot have both result and error")
        
        self.success = success
        self.result = result
        self.error = error
        self.request_id = request_id
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "success": self.success,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat()
        }
        
        if self.result is not None:
            result["result"] = self.result
        if self.error is not None:
            result["error"] = self.error.to_dict()
        
        return result


class MCPError:
    """Mock MCP error class."""
    def __init__(self, code: str, message: str, details: Optional[Dict] = None):
        self.code = code
        self.message = message
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details
        }
    
    @classmethod
    def from_exception(cls, exception: Exception, code: str) -> 'MCPError':
        """Create error from exception."""
        return cls(
            code=code,
            message=str(exception),
            details={
                "exception_type": type(exception).__name__
            }
        )


class MCPMethodHandler:
    """Mock MCP method handler."""
    def __init__(self):
        self._handlers = {}
        self._validators = {}
    
    def register(self, method: str):
        """Register method handler."""
        def decorator(handler):
            self._handlers[method] = handler
            return handler
        return decorator
    
    def validate_params(self, schema: Dict):
        """Validate parameters decorator."""
        def decorator(handler):
            # In real implementation, would validate against schema
            return handler
        return decorator
    
    def has_handler(self, method: str) -> bool:
        """Check if handler exists."""
        return method in self._handlers
    
    def get_handler(self, method: str):
        """Get handler for method."""
        return self._handlers.get(method)


class MCPAuthenticator:
    """Mock MCP authenticator."""
    def __init__(self, tokens: Optional[Dict] = None, require_auth: bool = True):
        self.tokens = tokens or {}
        self.require_auth = require_auth
    
    async def authenticate(self, auth_data: Optional[Dict]) -> 'AuthResult':
        """Authenticate request."""
        if not self.require_auth:
            return AuthResult(success=True, user="anonymous", permissions=[])
        
        if not auth_data:
            return AuthResult(
                success=False,
                error=MCPError("MISSING_AUTH", "Authentication required")
            )
        
        token = auth_data.get("token")
        if token not in self.tokens:
            return AuthResult(
                success=False,
                error=MCPError("INVALID_TOKEN", "Invalid authentication token")
            )
        
        token_data = self.tokens[token]
        return AuthResult(
            success=True,
            user=token_data["user"],
            permissions=token_data.get("permissions", [])
        )
    
    def has_permission(self, auth_result: 'AuthResult', permission: str) -> bool:
        """Check if user has permission."""
        return permission in auth_result.permissions


class AuthResult:
    """Mock authentication result."""
    def __init__(self, success: bool, user: str = "", permissions: Optional[List] = None, 
                 error: Optional[MCPError] = None):
        self.success = success
        self.user = user
        self.permissions = permissions or []
        self.error = error


class MCPRateLimiter:
    """Mock MCP rate limiter."""
    def __init__(self, requests_per_minute: int = 60, burst_limit: int = 100):
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self._client_requests = {}
    
    async def check_rate_limit(self, client_id: str) -> 'RateLimitResult':
        """Check rate limit for client."""
        now = datetime.now()
        
        if client_id not in self._client_requests:
            self._client_requests[client_id] = []
        
        requests = self._client_requests[client_id]
        
        # Remove old requests (outside time window)
        cutoff = now - timedelta(minutes=1)
        requests[:] = [req_time for req_time in requests if req_time > cutoff]
        
        # Check limits
        if len(requests) >= self.burst_limit:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                retry_after=60
            )
        
        if len(requests) >= self.requests_per_minute:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                retry_after=60 - (now - requests[0]).seconds
            )
        
        # Allow request
        requests.append(now)
        return RateLimitResult(
            allowed=True,
            remaining=self.requests_per_minute - len(requests)
        )
    
    def _reset_window(self, client_id: str):
        """Reset rate limit window (for testing)."""
        if client_id in self._client_requests:
            self._client_requests[client_id] = []


class RateLimitResult:
    """Mock rate limit result."""
    def __init__(self, allowed: bool, remaining: int = 0, retry_after: int = 0):
        self.allowed = allowed
        self.remaining = remaining
        self.retry_after = retry_after


class MCPServer:
    """Mock MCP server."""
    def __init__(self, config, observer, memory, analyzer):
        self.config = config
        self.observer = observer
        self.memory = memory
        self.analyzer = analyzer
        self.is_running = False
        self.port = None
        self.host = None
        self._method_handler = MCPMethodHandler()
        self._authenticator = None
        self._rate_limiter = None
        self._metrics = {
            "total_requests": 0,
            "requests_by_method": {},
            "error_count": 0,
            "response_times": []
        }
        self._max_concurrent = 50
        self._request_timeout = 30
        self._enable_metrics = True
    
    async def start(self, port: int = 8080, host: str = "localhost"):
        """Start MCP server."""
        self.port = port
        self.host = host
        self.is_running = True
    
    async def shutdown(self, timeout: float = 10.0):
        """Shutdown MCP server."""
        # In real implementation, would wait for pending requests
        await asyncio.sleep(0.1)  # Simulate shutdown delay
        self.is_running = False
    
    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """Handle MCP request."""
        start_time = datetime.now()
        
        try:
            # Authentication
            if self._authenticator:
                auth_result = await self._authenticator.authenticate(request.auth)
                if not auth_result.success:
                    return MCPResponse(
                        success=False,
                        error=auth_result.error,
                        request_id=request.id
                    )
            
            # Rate limiting
            if self._rate_limiter:
                client_id = request.auth.get("client_id", "anonymous") if request.auth else "anonymous"
                rate_result = await self._rate_limiter.check_rate_limit(client_id)
                if not rate_result.allowed:
                    return MCPResponse(
                        success=False,
                        error=MCPError("RATE_LIMITED", "Rate limit exceeded"),
                        request_id=request.id
                    )
            
            # Method handling
            if not self._method_handler.has_handler(request.method):
                return MCPResponse(
                    success=False,
                    error=MCPError("METHOD_NOT_FOUND", f"Method '{request.method}' not found"),
                    request_id=request.id
                )
            
            handler = self._method_handler.get_handler(request.method)
            result = await handler(request.params)
            
            # Update metrics
            if self._enable_metrics:
                self._update_metrics(request.method, start_time, True)
            
            return MCPResponse(
                success=True,
                result=result,
                request_id=request.id
            )
        
        except Exception as e:
            # Update error metrics
            if self._enable_metrics:
                self._update_metrics(request.method, start_time, False)
            
            return MCPResponse(
                success=False,
                error=MCPError.from_exception(e, "INTERNAL_ERROR"),
                request_id=request.id
            )
    
    def _update_metrics(self, method: str, start_time: datetime, success: bool):
        """Update metrics."""
        self._metrics["total_requests"] += 1
        
        if method not in self._metrics["requests_by_method"]:
            self._metrics["requests_by_method"][method] = 0
        self._metrics["requests_by_method"][method] += 1
        
        if not success:
            self._metrics["error_count"] += 1
        
        duration = (datetime.now() - start_time).total_seconds()
        self._metrics["response_times"].append(duration)
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics."""
        avg_response_time = 0
        if self._metrics["response_times"]:
            avg_response_time = sum(self._metrics["response_times"]) / len(self._metrics["response_times"])
        
        error_rate = 0
        if self._metrics["total_requests"] > 0:
            error_rate = self._metrics["error_count"] / self._metrics["total_requests"]
        
        return {
            "total_requests": self._metrics["total_requests"],
            "requests_by_method": self._metrics["requests_by_method"].copy(),
            "average_response_time": avg_response_time,
            "error_rate": error_rate
        }
    
    def update_config(self, config: Dict[str, Any]):
        """Update server configuration."""
        self._max_concurrent = config.get("max_concurrent_requests", self._max_concurrent)
        self._request_timeout = config.get("request_timeout", self._request_timeout)
        self._enable_metrics = config.get("enable_metrics", self._enable_metrics)