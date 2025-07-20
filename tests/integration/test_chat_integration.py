#!/usr/bin/env python3
"""
Integration tests for Chat functionality.

Tests the chat system integration including conversation management,
context retrieval, and LLM API interactions.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import chat-related components (mocked for testing)
# These imports will be adjusted based on actual implementation after migration
# from eidolon.core.enhanced_interface import EnhancedInterface, ChatSession, ChatMessage
from eidolon.core.memory import MemorySystem
from eidolon.models.cloud_api import CloudAPIManager, CloudAPIResponse


@pytest.mark.integration
@pytest.mark.asyncio
class TestChatIntegration:
    """Test chat functionality integration with Eidolon components."""
    
    @pytest.fixture
    async def chat_interface(self, mock_config, mock_memory_system, mock_cloud_api_manager):
        """Create enhanced interface with chat capabilities."""
        interface = EnhancedInterface(
            config=mock_config,
            memory=mock_memory_system,
            cloud_api=mock_cloud_api_manager
        )
        yield interface
        # Cleanup
        if hasattr(interface, 'cleanup'):
            await interface.cleanup()
    
    @pytest.fixture
    def mock_cloud_api_manager(self):
        """Mock CloudAPIManager for testing."""
        manager = AsyncMock(spec=CloudAPIManager)
        manager.analyze_with_context = AsyncMock()
        manager.get_available_providers = Mock(return_value=["gemini", "claude", "openai"])
        return manager
    
    @pytest.fixture
    def sample_chat_context(self):
        """Sample context for chat testing."""
        return {
            "recent_screenshots": [
                {
                    "id": "screen-1",
                    "timestamp": "2024-01-01T12:00:00",
                    "content": "User working on Python code",
                    "application": "VS Code"
                },
                {
                    "id": "screen-2",
                    "timestamp": "2024-01-01T12:05:00",
                    "content": "Running pytest tests",
                    "application": "Terminal"
                }
            ],
            "extracted_text": "import pytest\nclass TestExample:\n    def test_something(self):",
            "current_activity": "coding",
            "applications": ["VS Code", "Terminal", "Chrome"]
        }
    
    async def test_chat_initialization(self, chat_interface, mock_config):
        """Test chat system initialization with API connectivity."""
        # Initialize chat
        session = await chat_interface.create_chat_session()
        
        # Verify session creation
        assert session is not None
        assert session.id is not None
        assert session.created_at is not None
        assert len(session.messages) == 0
        assert session.context_window == mock_config.interface.chat.context_window
    
    async def test_chat_api_connectivity(self, chat_interface, mock_cloud_api_manager):
        """Test chat connectivity to LLM APIs."""
        # Test API availability
        available = await chat_interface.check_api_availability()
        
        assert available
        assert "gemini" in await chat_interface.get_available_providers()
        assert "claude" in await chat_interface.get_available_providers()
        
        # Verify cloud API manager was checked
        mock_cloud_api_manager.get_available_providers.assert_called()
    
    async def test_context_retrieval_from_screenshots(self, chat_interface, mock_memory_system, sample_chat_context):
        """Test context retrieval from screen captures for chat."""
        # Setup mock
        mock_memory_system.get_recent_context.return_value = sample_chat_context
        
        # Create chat session and send message
        session = await chat_interface.create_chat_session()
        
        # Get context for chat
        context = await chat_interface.get_chat_context(
            session_id=session.id,
            time_window_minutes=30
        )
        
        # Verify context
        assert context is not None
        assert len(context["recent_screenshots"]) == 2
        assert context["current_activity"] == "coding"
        assert "VS Code" in context["applications"]
        
        # Verify memory system was called
        mock_memory_system.get_recent_context.assert_called_once_with(
            time_window=timedelta(minutes=30)
        )
    
    async def test_conversation_flow(self, chat_interface, mock_cloud_api_manager, sample_chat_context):
        """Test multi-turn conversation handling."""
        # Setup mocks
        mock_responses = [
            CloudAPIResponse(
                content="I can see you're working on Python tests. How can I help?",
                model="gemini-pro",
                provider="gemini",
                confidence=0.95
            ),
            CloudAPIResponse(
                content="To add a new test, you can create a method starting with 'test_'",
                model="gemini-pro",
                provider="gemini",
                confidence=0.93
            ),
            CloudAPIResponse(
                content="Here's an example of a parametrized test using pytest.mark.parametrize",
                model="gemini-pro",
                provider="gemini",
                confidence=0.91
            )
        ]
        mock_cloud_api_manager.analyze_with_context.side_effect = mock_responses
        
        # Create session
        session = await chat_interface.create_chat_session()
        
        # First message
        response1 = await chat_interface.send_message(
            session_id=session.id,
            message="What am I working on?",
            include_context=True
        )
        
        assert response1.content == "I can see you're working on Python tests. How can I help?"
        assert len(session.messages) == 2  # User message + assistant response
        
        # Second message
        response2 = await chat_interface.send_message(
            session_id=session.id,
            message="How do I add a new test?",
            include_context=False
        )
        
        assert "test_" in response2.content
        assert len(session.messages) == 4
        
        # Third message
        response3 = await chat_interface.send_message(
            session_id=session.id,
            message="Show me an example with parametrize",
            include_context=False
        )
        
        assert "parametrize" in response3.content
        assert len(session.messages) == 6
        
        # Verify conversation history is maintained
        assert session.messages[0].role == "user"
        assert session.messages[1].role == "assistant"
        assert session.messages[2].content == "How do I add a new test?"
    
    async def test_conversation_history_management(self, chat_interface):
        """Test conversation history storage and retrieval."""
        # Create multiple sessions
        session1 = await chat_interface.create_chat_session()
        session2 = await chat_interface.create_chat_session()
        
        # Add messages to session1
        await chat_interface.send_message(
            session_id=session1.id,
            message="Hello from session 1"
        )
        
        # Add messages to session2
        await chat_interface.send_message(
            session_id=session2.id,
            message="Hello from session 2"
        )
        
        # Retrieve sessions
        sessions = await chat_interface.get_chat_sessions()
        assert len(sessions) >= 2
        
        # Retrieve specific session
        retrieved_session1 = await chat_interface.get_chat_session(session1.id)
        assert retrieved_session1.id == session1.id
        assert len(retrieved_session1.messages) == 2
        
        # Delete session
        await chat_interface.delete_chat_session(session2.id)
        sessions_after = await chat_interface.get_chat_sessions()
        assert len(sessions_after) == len(sessions) - 1
    
    async def test_memory_integration(self, chat_interface, mock_memory_system):
        """Test integration with existing memory systems."""
        # Setup mock memory search
        mock_memory_results = [
            {
                "id": "mem-1",
                "content": "Previously discussed Python testing patterns",
                "timestamp": "2024-01-01T10:00:00",
                "relevance": 0.85
            }
        ]
        mock_memory_system.search.return_value = mock_memory_results
        
        # Create session with memory integration
        session = await chat_interface.create_chat_session(
            enable_memory_search=True
        )
        
        # Send message that should trigger memory search
        response = await chat_interface.send_message(
            session_id=session.id,
            message="What have we discussed about testing before?",
            search_memory=True
        )
        
        # Verify memory was searched
        mock_memory_system.search.assert_called()
        call_args = mock_memory_system.search.call_args
        assert "testing" in call_args[0][0].lower()  # Query should contain "testing"
        
        # Response should incorporate memory results
        assert response.metadata.get("memory_results") is not None
        assert len(response.metadata["memory_results"]) == 1
    
    async def test_error_handling_and_fallbacks(self, chat_interface, mock_cloud_api_manager):
        """Test error handling and fallback mechanisms."""
        # Setup primary provider to fail
        mock_cloud_api_manager.analyze_with_context.side_effect = [
            Exception("Gemini API failed"),
            CloudAPIResponse(
                content="Response from fallback provider",
                model="claude-3",
                provider="claude",
                confidence=0.88
            )
        ]
        
        session = await chat_interface.create_chat_session()
        
        # Send message - should fallback to secondary provider
        response = await chat_interface.send_message(
            session_id=session.id,
            message="Test message",
            fallback_providers=["claude", "openai"]
        )
        
        # Verify fallback worked
        assert response.content == "Response from fallback provider"
        assert response.provider == "claude"
        assert mock_cloud_api_manager.analyze_with_context.call_count == 2
    
    async def test_streaming_responses(self, chat_interface, mock_cloud_api_manager):
        """Test streaming chat responses."""
        # Setup streaming mock
        async def mock_stream():
            chunks = ["Hello", " from", " streaming", " response"]
            for chunk in chunks:
                yield chunk
                await asyncio.sleep(0.01)
        
        mock_cloud_api_manager.stream_with_context = mock_stream
        
        session = await chat_interface.create_chat_session()
        
        # Collect streamed response
        chunks = []
        async for chunk in chat_interface.send_message_stream(
            session_id=session.id,
            message="Test streaming"
        ):
            chunks.append(chunk)
        
        # Verify streaming worked
        assert len(chunks) == 4
        assert "".join(chunks) == "Hello from streaming response"
    
    async def test_context_window_management(self, chat_interface, mock_config):
        """Test context window token management."""
        # Configure small context window for testing
        mock_config.interface.chat.context_window = 100
        mock_config.interface.chat.max_tokens = 50
        
        session = await chat_interface.create_chat_session()
        
        # Add messages until context window is exceeded
        for i in range(10):
            await chat_interface.send_message(
                session_id=session.id,
                message=f"This is a long message number {i} with lots of tokens"
            )
        
        # Verify old messages are pruned
        active_context = await chat_interface.get_active_context(session.id)
        assert active_context["token_count"] <= mock_config.interface.chat.context_window
        assert len(active_context["messages"]) < 20  # Less than all messages
    
    async def test_provider_specific_features(self, chat_interface, mock_cloud_api_manager):
        """Test provider-specific features and parameters."""
        session = await chat_interface.create_chat_session()
        
        # Test Gemini-specific features
        await chat_interface.send_message(
            session_id=session.id,
            message="Analyze this image",
            provider="gemini",
            provider_params={
                "temperature": 0.7,
                "top_p": 0.95,
                "safety_settings": "BLOCK_NONE"
            }
        )
        
        # Verify provider params were passed
        call_args = mock_cloud_api_manager.analyze_with_context.call_args
        assert call_args[1]["provider"] == "gemini"
        assert call_args[1]["temperature"] == 0.7
        
        # Test Claude-specific features
        await chat_interface.send_message(
            session_id=session.id,
            message="Complex reasoning task",
            provider="claude",
            provider_params={
                "max_tokens": 4096,
                "system": "You are a helpful assistant"
            }
        )
        
        # Verify Claude params
        call_args = mock_cloud_api_manager.analyze_with_context.call_args
        assert call_args[1]["provider"] == "claude"
        assert call_args[1]["max_tokens"] == 4096
    
    async def test_conversation_export_import(self, chat_interface):
        """Test conversation export and import functionality."""
        # Create session with messages
        session = await chat_interface.create_chat_session()
        
        await chat_interface.send_message(
            session_id=session.id,
            message="First message"
        )
        await chat_interface.send_message(
            session_id=session.id,
            message="Second message"
        )
        
        # Export conversation
        export_data = await chat_interface.export_conversation(
            session_id=session.id,
            format="json"
        )
        
        # Verify export
        assert export_data["session_id"] == session.id
        assert len(export_data["messages"]) == 4
        assert export_data["metadata"]["export_time"] is not None
        
        # Import conversation
        new_session = await chat_interface.import_conversation(
            export_data,
            create_new_session=True
        )
        
        # Verify import
        assert new_session.id != session.id
        assert len(new_session.messages) == 4
        assert new_session.messages[0].content == "First message"
    
    async def test_rate_limiting(self, chat_interface):
        """Test chat rate limiting to prevent API abuse."""
        session = await chat_interface.create_chat_session()
        
        # Configure rate limit
        chat_interface.configure_rate_limit(
            messages_per_minute=2,
            tokens_per_minute=100
        )
        
        # Send messages up to limit
        await chat_interface.send_message(session.id, "Message 1")
        await chat_interface.send_message(session.id, "Message 2")
        
        # Third message should be rate limited
        with pytest.raises(RateLimitError) as exc:
            await chat_interface.send_message(session.id, "Message 3")
        
        assert "rate limit" in str(exc.value).lower()
    
    async def test_multi_modal_chat(self, chat_interface, mock_cloud_api_manager):
        """Test multi-modal chat with images and text."""
        session = await chat_interface.create_chat_session()
        
        # Send message with image
        response = await chat_interface.send_message(
            session_id=session.id,
            message="What's in this screenshot?",
            attachments=[{
                "type": "image",
                "data": "base64_encoded_image_data",
                "mime_type": "image/png"
            }]
        )
        
        # Verify multi-modal handling
        call_args = mock_cloud_api_manager.analyze_with_context.call_args
        assert "attachments" in call_args[1]
        assert call_args[1]["attachments"][0]["type"] == "image"
    
    async def test_chat_with_tools(self, chat_interface, mock_cloud_api_manager):
        """Test chat with tool/function calling capabilities."""
        # Define available tools
        tools = [
            {
                "name": "search_memory",
                "description": "Search through stored memories",
                "parameters": {
                    "query": {"type": "string", "required": True},
                    "limit": {"type": "integer", "default": 10}
                }
            },
            {
                "name": "capture_screen",
                "description": "Capture current screen",
                "parameters": {
                    "area": {"type": "string", "enum": ["full", "window", "selection"]}
                }
            }
        ]
        
        session = await chat_interface.create_chat_session(
            enable_tools=True,
            available_tools=tools
        )
        
        # Mock tool call response
        mock_cloud_api_manager.analyze_with_context.return_value = CloudAPIResponse(
            content="I'll search for that information",
            model="gpt-4",
            provider="openai",
            confidence=0.9,
            tool_calls=[{
                "name": "search_memory",
                "arguments": {"query": "Python testing", "limit": 5}
            }]
        )
        
        # Send message that triggers tool use
        response = await chat_interface.send_message(
            session_id=session.id,
            message="Find my previous work on Python testing"
        )
        
        # Verify tool was called
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "search_memory"


@pytest.mark.integration
@pytest.mark.asyncio
class TestChatPersistence:
    """Test chat persistence and recovery."""
    
    @pytest.fixture
    async def persistent_chat(self, tmp_path, mock_config):
        """Create chat interface with persistence."""
        mock_config.interface.chat.persistence_path = str(tmp_path / "chat_sessions")
        
        from eidolon.core.enhanced_interface import PersistentChatInterface
        interface = PersistentChatInterface(config=mock_config)
        yield interface
        await interface.cleanup()
    
    async def test_session_persistence(self, persistent_chat):
        """Test chat sessions are persisted to disk."""
        # Create session
        session = await persistent_chat.create_chat_session()
        session_id = session.id
        
        # Add messages
        await persistent_chat.send_message(session_id, "Test message")
        
        # Force save
        await persistent_chat.save_sessions()
        
        # Create new instance and load
        new_chat = PersistentChatInterface(persistent_chat.config)
        await new_chat.load_sessions()
        
        # Verify session was loaded
        loaded_session = await new_chat.get_chat_session(session_id)
        assert loaded_session is not None
        assert len(loaded_session.messages) == 2
    
    async def test_crash_recovery(self, persistent_chat):
        """Test recovery from crashes."""
        session = await persistent_chat.create_chat_session()
        
        # Simulate crash by not properly closing
        # Messages should be auto-saved periodically
        await persistent_chat.send_message(session.id, "Message before crash")
        
        # Wait for auto-save
        await asyncio.sleep(0.1)
        
        # Simulate restart
        new_chat = PersistentChatInterface(persistent_chat.config)
        await new_chat.recover_sessions()
        
        # Verify recovery
        recovered = await new_chat.get_chat_session(session.id)
        assert recovered is not None
        assert any("before crash" in msg.content for msg in recovered.messages)


# Helper classes for testing
class ChatSession:
    """Mock chat session."""
    def __init__(self):
        self.id = f"session-{datetime.now().timestamp()}"
        self.created_at = datetime.now()
        self.messages: List[ChatMessage] = []
        self.context_window = 4096
        self.metadata = {}


class ChatMessage:
    """Mock chat message."""
    def __init__(self, role: str, content: str, **kwargs):
        self.role = role
        self.content = content
        self.timestamp = datetime.now()
        self.provider = kwargs.get("provider")
        self.confidence = kwargs.get("confidence")
        self.metadata = kwargs.get("metadata", {})
        self.tool_calls = kwargs.get("tool_calls")


class EnhancedInterface:
    """Mock enhanced interface with chat capabilities."""
    def __init__(self, config, memory, cloud_api):
        self.config = config
        self.memory = memory
        self.cloud_api = cloud_api
        self.sessions: Dict[str, ChatSession] = {}
        self._rate_limiter = None
    
    async def create_chat_session(self, **kwargs) -> ChatSession:
        """Create new chat session."""
        session = ChatSession()
        session.context_window = self.config.interface.chat.context_window
        if kwargs.get("enable_tools"):
            session.metadata["tools"] = kwargs.get("available_tools", [])
        self.sessions[session.id] = session
        return session
    
    async def check_api_availability(self) -> bool:
        """Check if APIs are available."""
        providers = await self.get_available_providers()
        return len(providers) > 0
    
    async def get_available_providers(self) -> List[str]:
        """Get available LLM providers."""
        return self.cloud_api.get_available_providers()
    
    async def get_chat_context(self, session_id: str, time_window_minutes: int) -> Dict:
        """Get context for chat."""
        return await self.memory.get_recent_context(
            time_window=timedelta(minutes=time_window_minutes)
        )
    
    async def send_message(self, session_id: str, message: str, **kwargs) -> ChatMessage:
        """Send message and get response."""
        session = self.sessions[session_id]
        
        # Add user message
        user_msg = ChatMessage("user", message)
        session.messages.append(user_msg)
        
        # Get response
        context = None
        if kwargs.get("include_context", True) or kwargs.get("search_memory"):
            context = await self.get_chat_context(session_id, 30)
        
        params = {
            "provider": kwargs.get("provider", "gemini"),
            "temperature": kwargs.get("provider_params", {}).get("temperature", 0.7),
            "max_tokens": kwargs.get("provider_params", {}).get("max_tokens", 1000)
        }
        
        if kwargs.get("attachments"):
            params["attachments"] = kwargs["attachments"]
        
        response = await self.cloud_api.analyze_with_context(
            prompt=message,
            context=context,
            **params
        )
        
        # Create assistant message
        assistant_msg = ChatMessage(
            "assistant",
            response.content,
            provider=response.provider,
            confidence=response.confidence,
            metadata={"memory_results": context.get("memory_results")} if context else {},
            tool_calls=getattr(response, "tool_calls", None)
        )
        session.messages.append(assistant_msg)
        
        return assistant_msg
    
    async def send_message_stream(self, session_id: str, message: str):
        """Stream message response."""
        async for chunk in self.cloud_api.stream_with_context(message):
            yield chunk
    
    async def get_chat_sessions(self) -> List[ChatSession]:
        """Get all chat sessions."""
        return list(self.sessions.values())
    
    async def get_chat_session(self, session_id: str) -> Optional[ChatSession]:
        """Get specific chat session."""
        return self.sessions.get(session_id)
    
    async def delete_chat_session(self, session_id: str):
        """Delete chat session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    async def get_active_context(self, session_id: str) -> Dict:
        """Get active context with token count."""
        session = self.sessions[session_id]
        # Mock token counting
        token_count = sum(len(msg.content.split()) * 1.3 for msg in session.messages)
        return {
            "messages": session.messages[-10:],  # Last 10 messages
            "token_count": int(token_count)
        }
    
    async def export_conversation(self, session_id: str, format: str) -> Dict:
        """Export conversation."""
        session = self.sessions[session_id]
        return {
            "session_id": session.id,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                }
                for msg in session.messages
            ],
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "format": format
            }
        }
    
    async def import_conversation(self, data: Dict, create_new_session: bool = True) -> ChatSession:
        """Import conversation."""
        session = ChatSession() if create_new_session else self.sessions.get(data["session_id"])
        
        for msg_data in data["messages"]:
            msg = ChatMessage(msg_data["role"], msg_data["content"])
            session.messages.append(msg)
        
        if create_new_session:
            self.sessions[session.id] = session
        
        return session
    
    def configure_rate_limit(self, messages_per_minute: int, tokens_per_minute: int):
        """Configure rate limiting."""
        self._rate_limiter = {
            "messages_per_minute": messages_per_minute,
            "tokens_per_minute": tokens_per_minute
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        self.sessions.clear()


class PersistentChatInterface(EnhancedInterface):
    """Mock persistent chat interface."""
    async def save_sessions(self):
        """Save sessions to disk."""
        # Mock save
        pass
    
    async def load_sessions(self):
        """Load sessions from disk."""
        # Mock load
        pass
    
    async def recover_sessions(self):
        """Recover sessions after crash."""
        # Mock recovery
        pass


class RateLimitError(Exception):
    """Rate limit error."""
    pass