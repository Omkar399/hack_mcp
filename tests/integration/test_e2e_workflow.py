#!/usr/bin/env python3
"""
End-to-end workflow integration tests.

Tests complete workflows from capture → analyze → query → respond,
including MCP server + chat integration and CLI integration.
"""

import pytest
import asyncio
import json
import tempfile
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

# Import core components (some mocked for testing)
from eidolon.core.observer import Observer
from eidolon.core.analyzer import Analyzer
from eidolon.core.memory import MemorySystem
# from eidolon.core.enhanced_interface import EnhancedInterface  # Will be implemented
# from eidolon.core.mcp_server import MCPServer  # Will be implemented
# from eidolon.cli.main import EidolonCLI  # Will be enhanced
# from eidolon.utils.config import Config  # Using mock_config fixture instead


@pytest.mark.integration
@pytest.mark.asyncio
class TestCompleteWorkflow:
    """Test complete Eidolon workflows end-to-end."""
    
    @pytest.fixture
    async def eidolon_system(self, mock_config, temp_storage_dir):
        """Create complete Eidolon system for testing."""
        # Create mocked components for testing
        observer = Mock(spec=Observer)
        memory = AsyncMock(spec=MemorySystem)
        analyzer = AsyncMock(spec=Analyzer)
        interface = Mock()  # EnhancedInterface will be mocked
        mcp_server = Mock()  # MCPServer will be mocked
        
        # Setup mock behaviors
        memory.initialize = AsyncMock()
        memory.close = AsyncMock()
        mcp_server.shutdown = AsyncMock()
        
        # Initialize system
        await memory.initialize()
        
        system = {
            "observer": observer,
            "memory": memory,
            "analyzer": analyzer,
            "interface": interface,
            "mcp_server": mcp_server,
            "config": mock_config
        }
        
        yield system
        
        # Cleanup
        await memory.close()
        await mcp_server.shutdown()
    
    async def test_capture_analyze_store_workflow(self, eidolon_system, sample_screenshot):
        """Test: capture → analyze → store workflow."""
        observer = eidolon_system["observer"]
        analyzer = eidolon_system["analyzer"]
        memory = eidolon_system["memory"]
        
        # Mock screenshot capture
        observer.capture_screenshot = Mock(return_value=sample_screenshot)
        
        # Mock analysis results
        mock_analysis = {
            "extracted_text": "import pytest\nclass TestExample:",
            "ui_elements": ["code_editor", "terminal"],
            "applications": ["VS Code"],
            "activities": ["coding", "testing"],
            "importance_score": 0.85,
            "summary": "User writing Python tests"
        }
        analyzer.analyze_screenshot = AsyncMock(return_value=mock_analysis)
        
        # Mock memory storage
        memory.store = AsyncMock(return_value="stored-123")
        
        # Execute workflow
        # 1. Capture
        screenshot = observer.capture_screenshot()
        assert screenshot == sample_screenshot
        
        # 2. Analyze
        analysis = await analyzer.analyze_screenshot(screenshot)
        assert analysis["importance_score"] == 0.85
        assert "pytest" in analysis["extracted_text"]
        
        # 3. Store
        storage_id = await memory.store(
            content={
                "screenshot": screenshot,
                "analysis": analysis
            },
            metadata={
                "timestamp": screenshot.timestamp,
                "type": "screenshot_analysis"
            }
        )
        assert storage_id == "stored-123"
        
        # Verify all components were called
        observer.capture_screenshot.assert_called_once()
        analyzer.analyze_screenshot.assert_called_once_with(screenshot)
        memory.store.assert_called_once()
    
    async def test_search_and_respond_workflow(self, eidolon_system):
        """Test: search → context → respond workflow."""
        memory = eidolon_system["memory"]
        interface = eidolon_system["interface"]
        
        # Mock search results
        mock_search_results = [
            {
                "id": "result-1",
                "content": "User was working on Python testing with pytest",
                "timestamp": "2024-01-01T12:00:00",
                "confidence": 0.92,
                "metadata": {"application": "VS Code", "activity": "coding"}
            },
            {
                "id": "result-2",
                "content": "Running test commands in terminal",
                "timestamp": "2024-01-01T12:05:00",
                "confidence": 0.88,
                "metadata": {"application": "Terminal", "activity": "testing"}
            }
        ]
        memory.search = AsyncMock(return_value=mock_search_results)
        
        # Mock interface response
        interface.cloud_api.analyze_with_context = AsyncMock()
        interface.cloud_api.analyze_with_context.return_value = Mock(
            content="Based on your recent activity, you were writing Python tests using pytest in VS Code and running them in the terminal.",
            provider="gemini",
            confidence=0.94
        )
        
        # Execute workflow
        # 1. Search
        results = await memory.search("what was I working on?")
        assert len(results) == 2
        assert "pytest" in results[0]["content"]
        
        # 2. Create chat session and query
        session = await interface.create_chat_session()
        response = await interface.send_message(
            session_id=session.id,
            message="What was I working on recently?",
            include_context=True
        )
        
        # 3. Verify response incorporates search results
        assert "Python tests" in response.content
        assert "pytest" in response.content
        assert response.provider == "gemini"
        
        # Verify components were called correctly
        memory.search.assert_called()
        interface.cloud_api.analyze_with_context.assert_called_once()
    
    async def test_mcp_chat_integration_workflow(self, eidolon_system):
        """Test MCP server working with chat functionality."""
        mcp_server = eidolon_system["mcp_server"]
        interface = eidolon_system["interface"]
        memory = eidolon_system["memory"]
        
        # Start MCP server
        await mcp_server.start(port=8080, host="localhost")
        
        # Create chat session
        session = await interface.create_chat_session()
        
        # Mock MCP request handling
        mock_mcp_response = {
            "success": True,
            "result": {
                "results": [
                    {
                        "id": "mcp-result-1",
                        "content": "Found coding activity from MCP search",
                        "timestamp": "2024-01-01T12:00:00"
                    }
                ]
            }
        }
        
        # Setup mock for MCP search through chat
        async def mock_mcp_search(query, **kwargs):
            return mock_mcp_response["result"]["results"]
        
        memory.search = mock_mcp_search
        interface.cloud_api.analyze_with_context = AsyncMock()
        interface.cloud_api.analyze_with_context.return_value = Mock(
            content="MCP search found your recent coding activity.",
            provider="claude",
            confidence=0.91
        )
        
        # Execute integrated workflow
        response = await interface.send_message(
            session_id=session.id,
            message="Search my recent activity via MCP",
            use_mcp_search=True
        )
        
        # Verify integration worked
        assert "MCP search" in response.content
        assert response.provider == "claude"
        
        # Cleanup
        await mcp_server.shutdown()
    
    async def test_full_pipeline_with_real_data_flow(self, eidolon_system, tmp_path):
        """Test full pipeline with realistic data flow."""
        observer = eidolon_system["observer"]
        analyzer = eidolon_system["analyzer"]
        memory = eidolon_system["memory"]
        interface = eidolon_system["interface"]
        
        # Create mock screenshot with real-like data
        screenshot_data = {
            "timestamp": "2024-01-01T14:30:00",
            "filename": "screenshot_1430.png",
            "width": 1920,
            "height": 1080,
            "file_path": str(tmp_path / "screenshot_1430.png"),
            "size_bytes": 2048000
        }
        
        # Mock realistic analysis
        analysis_data = {
            "extracted_text": """
            def test_user_authentication():
                user = User(email="test@example.com")
                assert user.authenticate("password123")
                
            def test_password_validation():
                validator = PasswordValidator()
                assert validator.is_strong("P@ssw0rd123")
            """,
            "ui_elements": [
                {"type": "code_editor", "bounds": [0, 0, 1200, 800]},
                {"type": "terminal", "bounds": [1200, 0, 1920, 400]},
                {"type": "browser", "bounds": [1200, 400, 1920, 1080]}
            ],
            "applications": ["VS Code", "Terminal", "Chrome"],
            "activities": ["coding", "testing", "research"],
            "topics": ["authentication", "password_validation", "security"],
            "importance_score": 0.89,
            "summary": "User developing authentication system with security testing",
            "confidence": 0.94
        }
        
        # Setup mocks with realistic data
        observer.capture_screenshot = Mock(return_value=Mock(**screenshot_data))
        analyzer.analyze_screenshot = AsyncMock(return_value=analysis_data)
        
        stored_items = []
        async def mock_store(content, metadata):
            item_id = f"item-{len(stored_items) + 1}"
            stored_items.append({
                "id": item_id,
                "content": content,
                "metadata": metadata,
                "stored_at": datetime.now()
            })
            return item_id
        
        async def mock_search(query, **kwargs):
            # Simple keyword matching for testing
            results = []
            keywords = query.lower().split()
            for item in stored_items:
                content_text = str(item["content"]).lower()
                if any(keyword in content_text for keyword in keywords):
                    results.append({
                        "id": item["id"],
                        "content": item["content"]["analysis"]["summary"],
                        "timestamp": item["metadata"]["timestamp"],
                        "confidence": 0.85,
                        "metadata": item["metadata"]
                    })
            return results
        
        memory.store = mock_store
        memory.search = mock_search
        
        # Execute full pipeline
        # 1. Capture and analyze multiple screenshots
        for i in range(3):
            screenshot = observer.capture_screenshot()
            analysis = await analyzer.analyze_screenshot(screenshot)
            
            await memory.store(
                content={
                    "screenshot": screenshot,
                    "analysis": analysis
                },
                metadata={
                    "timestamp": screenshot.timestamp,
                    "type": "screenshot_analysis",
                    "session": f"coding-session-{i}"
                }
            )
        
        # 2. Perform search
        search_results = await memory.search("authentication testing")
        assert len(search_results) == 3  # All items should match
        
        # 3. Create chat and ask about the data
        interface.cloud_api.analyze_with_context = AsyncMock()
        interface.cloud_api.analyze_with_context.return_value = Mock(
            content="You've been working on authentication system development with comprehensive security testing, including password validation and user authentication methods.",
            provider="gemini",
            confidence=0.92
        )
        
        session = await interface.create_chat_session()
        chat_response = await interface.send_message(
            session_id=session.id,
            message="What have I been working on with authentication?",
            include_context=True
        )
        
        # 4. Verify the full pipeline worked
        assert len(stored_items) == 3
        assert "authentication" in chat_response.content
        assert "security testing" in chat_response.content
        
        # Verify data flow integrity
        for item in stored_items:
            assert item["content"]["analysis"]["importance_score"] == 0.89
            assert "authentication" in item["content"]["analysis"]["topics"]
    
    async def test_error_recovery_in_pipeline(self, eidolon_system):
        """Test error recovery throughout the pipeline."""
        observer = eidolon_system["observer"]
        analyzer = eidolon_system["analyzer"]
        memory = eidolon_system["memory"]
        interface = eidolon_system["interface"]
        
        # Test observer failure recovery
        observer.capture_screenshot = Mock(side_effect=[
            Exception("Screenshot failed"),
            Mock(timestamp="2024-01-01T12:00:00", filename="recovery.png")
        ])
        
        # Should recover from first failure
        try:
            screenshot = observer.capture_screenshot()
            assert False, "Should have failed"
        except Exception as e:
            assert "Screenshot failed" in str(e)
        
        # Second attempt should succeed
        screenshot = observer.capture_screenshot()
        assert screenshot.filename == "recovery.png"
        
        # Test analyzer failure recovery
        analyzer.analyze_screenshot = AsyncMock(side_effect=[
            Exception("Analysis failed"),
            {"summary": "Recovery analysis", "importance_score": 0.5}
        ])
        
        try:
            analysis = await analyzer.analyze_screenshot(screenshot)
            assert False, "Should have failed"
        except Exception as e:
            assert "Analysis failed" in str(e)
        
        # Recovery attempt
        analysis = await analyzer.analyze_screenshot(screenshot)
        assert analysis["summary"] == "Recovery analysis"
        
        # Test memory failure recovery
        memory.store = AsyncMock(side_effect=[
            Exception("Storage failed"),
            "recovery-id-123"
        ])
        
        try:
            storage_id = await memory.store({"test": "data"}, {})
            assert False, "Should have failed"
        except Exception as e:
            assert "Storage failed" in str(e)
        
        # Recovery attempt
        storage_id = await memory.store({"test": "data"}, {})
        assert storage_id == "recovery-id-123"
    
    async def test_performance_under_load(self, eidolon_system):
        """Test system performance under load."""
        observer = eidolon_system["observer"]
        analyzer = eidolon_system["analyzer"]
        memory = eidolon_system["memory"]
        
        # Setup fast mocks
        observer.capture_screenshot = Mock(return_value=Mock(
            timestamp="2024-01-01T12:00:00",
            filename="load_test.png"
        ))
        
        analyzer.analyze_screenshot = AsyncMock(return_value={
            "summary": "Load test analysis",
            "importance_score": 0.7
        })
        
        storage_counter = 0
        async def fast_store(content, metadata):
            nonlocal storage_counter
            storage_counter += 1
            return f"load-item-{storage_counter}"
        
        memory.store = fast_store
        
        # Execute concurrent operations
        start_time = datetime.now()
        
        tasks = []
        for i in range(10):
            async def process_item(item_id):
                screenshot = observer.capture_screenshot()
                analysis = await analyzer.analyze_screenshot(screenshot)
                storage_id = await memory.store(
                    {"screenshot": screenshot, "analysis": analysis},
                    {"timestamp": screenshot.timestamp, "item_id": item_id}
                )
                return storage_id
            
            tasks.append(process_item(i))
        
        results = await asyncio.gather(*tasks)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Verify performance
        assert len(results) == 10
        assert duration < 5.0  # Should complete in under 5 seconds
        assert storage_counter == 10
        
        # Verify all operations completed successfully
        for i, result in enumerate(results):
            assert result == f"load-item-{i + 1}"
    
    async def test_configuration_validation_workflow(self, eidolon_system):
        """Test configuration validation throughout workflow."""
        config = eidolon_system["config"]
        
        # Test observer configuration
        assert hasattr(config.observer, "capture_interval")
        assert hasattr(config.observer, "storage_path")
        assert config.observer.capture_interval > 0
        
        # Test analysis configuration
        assert hasattr(config.analysis, "local_models")
        assert hasattr(config.analysis, "cloud_apis")
        assert hasattr(config.analysis, "routing")
        
        # Test memory configuration
        assert hasattr(config.memory, "vector_db")
        assert hasattr(config.memory, "chunk_size")
        assert config.memory.chunk_size > 0
        
        # Test interface configuration
        assert hasattr(config.interface, "api_port")
        assert hasattr(config.interface, "web_port")
        assert config.interface.api_port != config.interface.web_port
        
        # Test privacy configuration
        assert hasattr(config.privacy, "local_only_mode")
        assert hasattr(config.privacy, "excluded_apps")
        assert isinstance(config.privacy.excluded_apps, list)


@pytest.mark.integration
@pytest.mark.asyncio
class TestCLIIntegration:
    """Test CLI integration with new features."""
    
    @pytest.fixture
    def cli_app(self, mock_config, tmp_path):
        """Create CLI app for testing."""
        # Setup test environment
        mock_config.observer.storage_path = str(tmp_path / "screenshots")
        mock_config.memory.db_path = str(tmp_path / "test.db")
        
        cli = EidolonCLI(config=mock_config)
        return cli
    
    async def test_cli_capture_command(self, cli_app):
        """Test CLI capture command with new features."""
        # Mock the observer
        with patch.object(cli_app, 'observer') as mock_observer:
            mock_observer.capture_screenshot.return_value = Mock(
                filename="cli_test.png",
                timestamp="2024-01-01T12:00:00"
            )
            
            # Execute capture command
            result = await cli_app.capture(
                area="full",
                analyze=True,
                store=True
            )
            
            # Verify result
            assert result["success"]
            assert result["screenshot"]["filename"] == "cli_test.png"
            assert "analysis" in result
            
            mock_observer.capture_screenshot.assert_called_once()
    
    async def test_cli_search_command(self, cli_app):
        """Test CLI search command."""
        # Mock memory system
        with patch.object(cli_app, 'memory') as mock_memory:
            mock_memory.search.return_value = [
                {
                    "id": "cli-result-1",
                    "content": "CLI search result",
                    "timestamp": "2024-01-01T12:00:00",
                    "confidence": 0.9
                }
            ]
            
            # Execute search command
            results = await cli_app.search(
                query="test query",
                limit=10,
                format="json"
            )
            
            # Verify results
            assert len(results) == 1
            assert results[0]["content"] == "CLI search result"
            
            mock_memory.search.assert_called_once_with(
                query="test query",
                limit=10
            )
    
    async def test_cli_chat_command(self, cli_app):
        """Test CLI chat command integration."""
        # Mock interface
        with patch.object(cli_app, 'interface') as mock_interface:
            mock_session = Mock(id="cli-session-123")
            mock_interface.create_chat_session.return_value = mock_session
            
            mock_response = Mock(
                content="CLI chat response",
                provider="gemini",
                confidence=0.88
            )
            mock_interface.send_message.return_value = mock_response
            
            # Execute chat command
            response = await cli_app.chat(
                message="Hello from CLI",
                provider="gemini",
                include_context=True
            )
            
            # Verify response
            assert response["content"] == "CLI chat response"
            assert response["provider"] == "gemini"
            
            mock_interface.create_chat_session.assert_called_once()
            mock_interface.send_message.assert_called_once()
    
    async def test_cli_mcp_server_command(self, cli_app):
        """Test CLI MCP server management."""
        # Mock MCP server
        with patch.object(cli_app, 'mcp_server') as mock_mcp:
            mock_mcp.is_running = False
            mock_mcp.start = AsyncMock()
            mock_mcp.shutdown = AsyncMock()
            
            # Test start command
            result = await cli_app.start_mcp_server(
                port=8080,
                host="localhost"
            )
            
            assert result["success"]
            assert result["status"] == "started"
            mock_mcp.start.assert_called_once_with(port=8080, host="localhost")
            
            # Test status command
            mock_mcp.is_running = True
            status = await cli_app.mcp_server_status()
            
            assert status["running"]
            assert status["port"] == 8080
            
            # Test stop command
            result = await cli_app.stop_mcp_server()
            
            assert result["success"]
            assert result["status"] == "stopped"
            mock_mcp.shutdown.assert_called_once()
    
    async def test_cli_config_validation(self, cli_app):
        """Test CLI configuration validation."""
        # Test config show command
        config_data = cli_app.show_config()
        
        # Verify configuration structure
        assert "observer" in config_data
        assert "analysis" in config_data
        assert "memory" in config_data
        assert "interface" in config_data
        assert "privacy" in config_data
        
        # Test config validation
        validation_result = cli_app.validate_config()
        
        assert validation_result["valid"]
        assert "errors" in validation_result
        assert len(validation_result["errors"]) == 0
    
    async def test_cli_export_import(self, cli_app, tmp_path):
        """Test CLI data export/import functionality."""
        export_path = tmp_path / "export.json"
        
        # Mock data for export
        with patch.object(cli_app, 'memory') as mock_memory:
            mock_data = {
                "sessions": [
                    {"id": "session-1", "messages": ["Hello", "Hi there"]},
                    {"id": "session-2", "messages": ["Test", "Response"]}
                ],
                "screenshots": [
                    {"id": "screen-1", "timestamp": "2024-01-01T12:00:00"}
                ]
            }
            mock_memory.export_data.return_value = mock_data
            
            # Test export
            result = await cli_app.export_data(
                output_path=str(export_path),
                format="json",
                include_screenshots=True
            )
            
            assert result["success"]
            assert result["exported_items"] == 3  # 2 sessions + 1 screenshot
            
            # Test import
            mock_memory.import_data = AsyncMock(return_value={
                "imported_items": 3,
                "skipped_items": 0
            })
            
            import_result = await cli_app.import_data(
                input_path=str(export_path),
                overwrite=False
            )
            
            assert import_result["success"]
            assert import_result["imported_items"] == 3


# Helper mock classes
class EidolonCLI:
    """Mock CLI interface."""
    def __init__(self, config):
        self.config = config
        self.observer = Mock()
        self.memory = Mock()
        self.interface = Mock()
        self.mcp_server = Mock()
    
    async def capture(self, area="full", analyze=False, store=False):
        """Mock capture command."""
        screenshot = self.observer.capture_screenshot()
        result = {
            "success": True,
            "screenshot": {
                "filename": screenshot.filename,
                "timestamp": screenshot.timestamp
            }
        }
        
        if analyze:
            result["analysis"] = {
                "summary": "Mock analysis",
                "importance_score": 0.8
            }
        
        return result
    
    async def search(self, query, limit=10, format="json"):
        """Mock search command."""
        return await self.memory.search(query=query, limit=limit)
    
    async def chat(self, message, provider="gemini", include_context=True):
        """Mock chat command."""
        session = await self.interface.create_chat_session()
        response = await self.interface.send_message(
            session_id=session.id,
            message=message,
            provider=provider,
            include_context=include_context
        )
        
        return {
            "content": response.content,
            "provider": response.provider,
            "confidence": response.confidence
        }
    
    async def start_mcp_server(self, port=8080, host="localhost"):
        """Mock MCP server start."""
        await self.mcp_server.start(port=port, host=host)
        return {"success": True, "status": "started", "port": port}
    
    async def mcp_server_status(self):
        """Mock MCP server status."""
        return {
            "running": self.mcp_server.is_running,
            "port": 8080,
            "host": "localhost"
        }
    
    async def stop_mcp_server(self):
        """Mock MCP server stop."""
        await self.mcp_server.shutdown()
        return {"success": True, "status": "stopped"}
    
    def show_config(self):
        """Mock config display."""
        return {
            "observer": {"capture_interval": 10},
            "analysis": {"local_models": {}},
            "memory": {"vector_db": "chromadb"},
            "interface": {"api_port": 8000},
            "privacy": {"local_only_mode": False}
        }
    
    def validate_config(self):
        """Mock config validation."""
        return {"valid": True, "errors": []}
    
    async def export_data(self, output_path, format="json", include_screenshots=True):
        """Mock data export."""
        exported = await self.memory.export_data(
            format=format,
            include_screenshots=include_screenshots
        )
        return {
            "success": True,
            "exported_items": len(exported.get("sessions", [])) + len(exported.get("screenshots", []))
        }
    
    async def import_data(self, input_path, overwrite=False):
        """Mock data import."""
        result = await self.memory.import_data(
            input_path=input_path,
            overwrite=overwrite
        )
        return {
            "success": True,
            "imported_items": result["imported_items"]
        }