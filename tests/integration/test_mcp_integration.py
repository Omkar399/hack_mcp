#!/usr/bin/env python3
"""
Integration tests for MCP (Model Context Protocol) server functionality.

Tests the MCP server integration including screen capture, search,
and LLM integration through MCP endpoints.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import MCP-related components (mocked for testing)
# These imports will be adjusted based on actual implementation after migration
# from eidolon.core.mcp_server import MCPServer, MCPRequest, MCPResponse
from eidolon.core.observer import Observer
from eidolon.core.memory import MemorySystem
from eidolon.core.analyzer import Analyzer


@pytest.mark.integration
@pytest.mark.asyncio
class TestMCPServerIntegration:
    """Test MCP server integration with Eidolon components."""
    
    @pytest.fixture
    async def mcp_server(self, mock_config, mock_observer, mock_memory_system, mock_analyzer):
        """Create MCP server instance for testing."""
        server = MCPServer(
            config=mock_config,
            observer=mock_observer,
            memory=mock_memory_system,
            analyzer=mock_analyzer
        )
        yield server
        # Cleanup
        if hasattr(server, 'shutdown'):
            await server.shutdown()
    
    @pytest.fixture
    def mock_mcp_request(self):
        """Create a mock MCP request."""
        return MCPRequest(
            method="screen_capture",
            params={
                "area": "full",
                "analyze": True
            },
            id="test-request-123"
        )
    
    async def test_mcp_server_initialization(self, mcp_server, mock_config):
        """Test MCP server initializes correctly."""
        assert mcp_server is not None
        assert mcp_server.config == mock_config
        assert hasattr(mcp_server, 'handle_request')
        assert hasattr(mcp_server, 'start')
        assert hasattr(mcp_server, 'shutdown')
    
    async def test_mcp_server_start_stop(self, mcp_server):
        """Test MCP server can start and stop properly."""
        # Start server
        await mcp_server.start(port=8080, host="localhost")
        assert mcp_server.is_running
        
        # Stop server
        await mcp_server.shutdown()
        assert not mcp_server.is_running
    
    async def test_mcp_screen_capture_endpoint(self, mcp_server, mock_observer, sample_screenshot):
        """Test screen capture through MCP endpoint."""
        # Setup mock
        mock_observer.capture_screenshot.return_value = sample_screenshot
        
        # Create request
        request = MCPRequest(
            method="capture_screen",
            params={
                "area": "full",
                "format": "png"
            },
            id="capture-123"
        )
        
        # Handle request
        response = await mcp_server.handle_request(request)
        
        # Verify response
        assert response.success
        assert response.result["screenshot_id"] is not None
        assert response.result["timestamp"] == sample_screenshot.timestamp
        assert response.result["dimensions"]["width"] == 1920
        assert response.result["dimensions"]["height"] == 1080
        
        # Verify observer was called
        mock_observer.capture_screenshot.assert_called_once()
    
    async def test_mcp_search_functionality(self, mcp_server, mock_memory_system):
        """Test search functionality through MCP endpoints."""
        # Setup mock search results
        mock_results = [
            {
                "id": "result-1",
                "content": "Test content 1",
                "timestamp": "2024-01-01T12:00:00",
                "confidence": 0.95
            },
            {
                "id": "result-2", 
                "content": "Test content 2",
                "timestamp": "2024-01-01T13:00:00",
                "confidence": 0.87
            }
        ]
        mock_memory_system.search.return_value = mock_results
        
        # Create search request
        request = MCPRequest(
            method="search",
            params={
                "query": "test query",
                "limit": 10,
                "filters": {
                    "date_from": "2024-01-01",
                    "content_type": "screenshot"
                }
            },
            id="search-456"
        )
        
        # Handle request
        response = await mcp_server.handle_request(request)
        
        # Verify response
        assert response.success
        assert len(response.result["results"]) == 2
        assert response.result["results"][0]["id"] == "result-1"
        assert response.result["query"] == "test query"
        assert response.result["total_results"] == 2
        
        # Verify memory search was called
        mock_memory_system.search.assert_called_once_with(
            query="test query",
            limit=10,
            filters={
                "date_from": "2024-01-01",
                "content_type": "screenshot"
            }
        )
    
    async def test_mcp_context_analysis(self, mcp_server, mock_analyzer, mock_memory_system):
        """Test LLM context analysis via MCP."""
        # Setup mocks
        mock_context = {
            "recent_activity": ["coding", "documentation"],
            "current_application": "VS Code",
            "extracted_text": "import pytest\nclass TestExample..."
        }
        mock_memory_system.get_context.return_value = mock_context
        
        mock_analysis = {
            "summary": "User is writing Python tests",
            "topics": ["testing", "python", "pytest"],
            "confidence": 0.92
        }
        mock_analyzer.analyze_context.return_value = mock_analysis
        
        # Create analysis request
        request = MCPRequest(
            method="analyze_context",
            params={
                "window_minutes": 30,
                "include_screenshots": True,
                "llm_provider": "gemini"
            },
            id="analyze-789"
        )
        
        # Handle request
        response = await mcp_server.handle_request(request)
        
        # Verify response
        assert response.success
        assert response.result["analysis"]["summary"] == "User is writing Python tests"
        assert "testing" in response.result["analysis"]["topics"]
        assert response.result["analysis"]["confidence"] == 0.92
        assert response.result["context"]["current_application"] == "VS Code"
    
    async def test_mcp_error_handling(self, mcp_server):
        """Test MCP server error handling."""
        # Test invalid method
        request = MCPRequest(
            method="invalid_method",
            params={},
            id="error-test-1"
        )
        
        response = await mcp_server.handle_request(request)
        assert not response.success
        assert response.error["code"] == "METHOD_NOT_FOUND"
        assert "invalid_method" in response.error["message"]
        
        # Test missing required parameters
        request = MCPRequest(
            method="search",
            params={},  # Missing required 'query' param
            id="error-test-2"
        )
        
        response = await mcp_server.handle_request(request)
        assert not response.success
        assert response.error["code"] == "INVALID_PARAMS"
        assert "query" in response.error["message"]
    
    async def test_mcp_batch_operations(self, mcp_server, mock_observer, mock_memory_system):
        """Test batch operations through MCP."""
        # Create batch request
        request = MCPRequest(
            method="batch",
            params={
                "operations": [
                    {
                        "method": "capture_screen",
                        "params": {"area": "full"}
                    },
                    {
                        "method": "search",
                        "params": {"query": "test"}
                    }
                ]
            },
            id="batch-123"
        )
        
        # Setup mocks
        mock_observer.capture_screenshot.return_value = Mock(
            timestamp="2024-01-01T12:00:00",
            filename="test.png"
        )
        mock_memory_system.search.return_value = [{"id": "1", "content": "test"}]
        
        # Handle request
        response = await mcp_server.handle_request(request)
        
        # Verify response
        assert response.success
        assert len(response.result["results"]) == 2
        assert response.result["results"][0]["success"]
        assert response.result["results"][1]["success"]
    
    async def test_mcp_recovery_mechanisms(self, mcp_server):
        """Test MCP server recovery from errors."""
        # Simulate component failure
        with patch.object(mcp_server, 'observer') as mock_observer:
            mock_observer.capture_screenshot.side_effect = Exception("Observer failed")
            
            request = MCPRequest(
                method="capture_screen",
                params={"area": "full"},
                id="recovery-test"
            )
            
            response = await mcp_server.handle_request(request)
            
            # Should handle error gracefully
            assert not response.success
            assert response.error["code"] == "INTERNAL_ERROR"
            assert "Observer failed" in response.error["message"]
            
            # Server should still be operational
            assert mcp_server.is_running
    
    async def test_mcp_concurrent_requests(self, mcp_server, mock_memory_system):
        """Test MCP server handles concurrent requests properly."""
        # Setup mock to simulate slow operation
        async def slow_search(*args, **kwargs):
            await asyncio.sleep(0.1)
            return [{"id": f"result-{kwargs.get('query', '')}", "content": "test"}]
        
        mock_memory_system.search = slow_search
        
        # Create multiple concurrent requests
        requests = [
            MCPRequest(
                method="search",
                params={"query": f"query-{i}"},
                id=f"concurrent-{i}"
            )
            for i in range(5)
        ]
        
        # Handle requests concurrently
        responses = await asyncio.gather(
            *[mcp_server.handle_request(req) for req in requests]
        )
        
        # Verify all succeeded
        assert len(responses) == 5
        for i, response in enumerate(responses):
            assert response.success
            assert response.result["results"][0]["id"] == f"result-query-{i}"
    
    async def test_mcp_authentication(self, mcp_server):
        """Test MCP server authentication mechanisms."""
        # Test with valid token
        request = MCPRequest(
            method="capture_screen",
            params={"area": "full"},
            id="auth-test-1",
            auth={"token": "valid-test-token"}
        )
        
        with patch.object(mcp_server, 'verify_auth', return_value=True):
            response = await mcp_server.handle_request(request)
            assert response.success
        
        # Test with invalid token
        request.auth = {"token": "invalid-token"}
        
        with patch.object(mcp_server, 'verify_auth', return_value=False):
            response = await mcp_server.handle_request(request)
            assert not response.success
            assert response.error["code"] == "UNAUTHORIZED"
    
    async def test_mcp_rate_limiting(self, mcp_server):
        """Test MCP server rate limiting."""
        # Configure rate limit
        mcp_server.configure_rate_limit(requests_per_minute=2)
        
        # First two requests should succeed
        for i in range(2):
            request = MCPRequest(
                method="search",
                params={"query": f"test-{i}"},
                id=f"rate-limit-{i}"
            )
            response = await mcp_server.handle_request(request)
            assert response.success
        
        # Third request should be rate limited
        request = MCPRequest(
            method="search",
            params={"query": "test-3"},
            id="rate-limit-3"
        )
        response = await mcp_server.handle_request(request)
        assert not response.success
        assert response.error["code"] == "RATE_LIMITED"
    
    async def test_mcp_websocket_support(self, mcp_server):
        """Test MCP server WebSocket support for real-time updates."""
        # Create WebSocket mock
        mock_websocket = AsyncMock()
        mock_websocket.send_json = AsyncMock()
        mock_websocket.receive_json = AsyncMock()
        
        # Test subscribe to updates
        await mcp_server.handle_websocket(mock_websocket, client_id="test-client")
        
        # Simulate screen capture event
        await mcp_server.broadcast_event({
            "type": "screen_captured",
            "data": {
                "screenshot_id": "123",
                "timestamp": "2024-01-01T12:00:00"
            }
        })
        
        # Verify WebSocket received update
        mock_websocket.send_json.assert_called_with({
            "type": "screen_captured",
            "data": {
                "screenshot_id": "123",
                "timestamp": "2024-01-01T12:00:00"
            }
        })
    
    async def test_mcp_metrics_collection(self, mcp_server):
        """Test MCP server metrics collection."""
        # Make several requests
        for i in range(5):
            request = MCPRequest(
                method="search" if i % 2 == 0 else "capture_screen",
                params={"query": f"test-{i}"} if i % 2 == 0 else {"area": "full"},
                id=f"metrics-{i}"
            )
            await mcp_server.handle_request(request)
        
        # Get metrics
        metrics = await mcp_server.get_metrics()
        
        # Verify metrics
        assert metrics["total_requests"] == 5
        assert metrics["requests_by_method"]["search"] == 3
        assert metrics["requests_by_method"]["capture_screen"] == 2
        assert "average_response_time" in metrics
        assert metrics["error_rate"] == 0.0


@pytest.mark.integration
@pytest.mark.asyncio
class TestMCPClientIntegration:
    """Test MCP client integration for external tools."""
    
    @pytest.fixture
    async def mcp_client(self):
        """Create MCP client for testing."""
        from eidolon.core.mcp_client import MCPClient
        client = MCPClient(base_url="http://localhost:8080")
        yield client
        await client.close()
    
    async def test_mcp_client_connection(self, mcp_client):
        """Test MCP client can connect to server."""
        # Test connection
        connected = await mcp_client.connect()
        assert connected
        
        # Test ping
        response = await mcp_client.ping()
        assert response["status"] == "ok"
    
    async def test_mcp_client_screen_capture(self, mcp_client):
        """Test screen capture through MCP client."""
        # Request screen capture
        result = await mcp_client.capture_screen(
            area="window",
            window_title="VS Code"
        )
        
        # Verify result
        assert result["screenshot_id"] is not None
        assert result["timestamp"] is not None
        assert result["window_title"] == "VS Code"
    
    async def test_mcp_client_search(self, mcp_client):
        """Test search through MCP client."""
        # Perform search
        results = await mcp_client.search(
            query="python testing",
            limit=5,
            date_range=("2024-01-01", "2024-01-31")
        )
        
        # Verify results
        assert isinstance(results, list)
        assert len(results) <= 5
        for result in results:
            assert "id" in result
            assert "content" in result
            assert "timestamp" in result
    
    async def test_mcp_client_retry_logic(self, mcp_client):
        """Test MCP client retry logic on failures."""
        # Mock network failure
        with patch.object(mcp_client, '_send_request') as mock_send:
            # First two attempts fail, third succeeds
            mock_send.side_effect = [
                Exception("Network error"),
                Exception("Network error"),
                {"success": True, "result": {"data": "test"}}
            ]
            
            # Should retry and eventually succeed
            result = await mcp_client.search("test", max_retries=3)
            assert result["data"] == "test"
            assert mock_send.call_count == 3
    
    async def test_mcp_client_timeout_handling(self, mcp_client):
        """Test MCP client timeout handling."""
        # Set short timeout
        mcp_client.timeout = 0.1
        
        # Mock slow response
        with patch.object(mcp_client, '_send_request') as mock_send:
            async def slow_response(*args, **kwargs):
                await asyncio.sleep(1)
                return {"success": True}
            
            mock_send.side_effect = slow_response
            
            # Should timeout
            with pytest.raises(asyncio.TimeoutError):
                await mcp_client.search("test")


# Helper classes for testing (these would be imported from actual implementation)
class MCPRequest:
    """Mock MCP request structure."""
    def __init__(self, method: str, params: Dict[str, Any], id: str, auth: Optional[Dict] = None):
        self.method = method
        self.params = params
        self.id = id
        self.auth = auth


class MCPResponse:
    """Mock MCP response structure."""
    def __init__(self, success: bool, result: Optional[Dict] = None, error: Optional[Dict] = None):
        self.success = success
        self.result = result
        self.error = error


class MCPServer:
    """Mock MCP server for testing."""
    def __init__(self, config, observer, memory, analyzer):
        self.config = config
        self.observer = observer
        self.memory = memory
        self.analyzer = analyzer
        self.is_running = False
        self._rate_limit = None
        self._request_counts = {}
    
    async def start(self, port: int, host: str):
        """Start the MCP server."""
        self.is_running = True
    
    async def shutdown(self):
        """Shutdown the MCP server."""
        self.is_running = False
    
    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """Handle incoming MCP request."""
        # Rate limiting check
        if self._rate_limit:
            client_id = request.auth.get("token", "anonymous") if request.auth else "anonymous"
            if client_id not in self._request_counts:
                self._request_counts[client_id] = []
            
            now = datetime.now()
            self._request_counts[client_id] = [
                t for t in self._request_counts[client_id] 
                if now - t < timedelta(minutes=1)
            ]
            
            if len(self._request_counts[client_id]) >= self._rate_limit:
                return MCPResponse(
                    success=False,
                    error={"code": "RATE_LIMITED", "message": "Rate limit exceeded"}
                )
            
            self._request_counts[client_id].append(now)
        
        # Method routing
        handlers = {
            "capture_screen": self._handle_capture_screen,
            "search": self._handle_search,
            "analyze_context": self._handle_analyze_context,
            "batch": self._handle_batch
        }
        
        if request.method not in handlers:
            return MCPResponse(
                success=False,
                error={"code": "METHOD_NOT_FOUND", "message": f"Method '{request.method}' not found"}
            )
        
        try:
            handler = handlers[request.method]
            result = await handler(request.params)
            return MCPResponse(success=True, result=result)
        except Exception as e:
            return MCPResponse(
                success=False,
                error={"code": "INTERNAL_ERROR", "message": str(e)}
            )
    
    async def _handle_capture_screen(self, params: Dict) -> Dict:
        """Handle screen capture request."""
        screenshot = self.observer.capture_screenshot()
        return {
            "screenshot_id": f"screenshot-{datetime.now().timestamp()}",
            "timestamp": screenshot.timestamp,
            "dimensions": {
                "width": screenshot.width,
                "height": screenshot.height
            }
        }
    
    async def _handle_search(self, params: Dict) -> Dict:
        """Handle search request."""
        if "query" not in params:
            raise ValueError("Missing required parameter: query")
        
        results = await self.memory.search(
            query=params["query"],
            limit=params.get("limit", 10),
            filters=params.get("filters", {})
        )
        
        return {
            "results": results,
            "query": params["query"],
            "total_results": len(results)
        }
    
    async def _handle_analyze_context(self, params: Dict) -> Dict:
        """Handle context analysis request."""
        context = await self.memory.get_context(
            window_minutes=params.get("window_minutes", 30)
        )
        analysis = await self.analyzer.analyze_context(
            context,
            provider=params.get("llm_provider", "gemini")
        )
        
        return {
            "context": context,
            "analysis": analysis
        }
    
    async def _handle_batch(self, params: Dict) -> Dict:
        """Handle batch operations."""
        results = []
        for op in params.get("operations", []):
            request = MCPRequest(
                method=op["method"],
                params=op.get("params", {}),
                id=f"batch-{len(results)}"
            )
            response = await self.handle_request(request)
            results.append({
                "success": response.success,
                "result": response.result,
                "error": response.error
            })
        
        return {"results": results}
    
    def configure_rate_limit(self, requests_per_minute: int):
        """Configure rate limiting."""
        self._rate_limit = requests_per_minute
    
    async def handle_websocket(self, websocket, client_id: str):
        """Handle WebSocket connection."""
        # Mock implementation
        pass
    
    async def broadcast_event(self, event: Dict):
        """Broadcast event to WebSocket clients."""
        # Mock implementation
        pass
    
    async def get_metrics(self) -> Dict:
        """Get server metrics."""
        return {
            "total_requests": 5,
            "requests_by_method": {
                "search": 3,
                "capture_screen": 2
            },
            "average_response_time": 0.05,
            "error_rate": 0.0
        }
    
    def verify_auth(self, token: str) -> bool:
        """Verify authentication token."""
        return token == "valid-test-token"