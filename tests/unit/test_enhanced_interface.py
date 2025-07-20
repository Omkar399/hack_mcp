#!/usr/bin/env python3
"""
Unit tests for Enhanced Interface component.

Tests the enhanced interface implementation including chat functionality,
context management, and provider integration.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Import enhanced interface components (mocked for testing)
# These would be imported from actual implementation after migration
# from eidolon.core.enhanced_interface import (
#     EnhancedInterface, ChatSession, ChatMessage, ConversationHistory,
#     ContextManager, ChatConfiguration, RateLimitError
# )


class TestChatMessage:
    """Test ChatMessage class."""
    
    def test_chat_message_creation(self):
        """Test ChatMessage creation with all fields."""
        message = ChatMessage(
            role="user",
            content="Hello, how are you?",
            provider="gemini",
            confidence=0.95,
            metadata={"source": "test", "tokens": 100},
            tool_calls=[{"name": "search", "args": {"query": "test"}}]
        )
        
        assert message.role == "user"
        assert message.content == "Hello, how are you?"
        assert message.provider == "gemini"
        assert message.confidence == 0.95
        assert message.metadata["source"] == "test"
        assert message.metadata["tokens"] == 100
        assert len(message.tool_calls) == 1
        assert isinstance(message.timestamp, datetime)
        assert isinstance(message.id, str)
    
    def test_chat_message_defaults(self):
        """Test ChatMessage with default values."""
        message = ChatMessage(
            role="assistant",
            content="Default message"
        )
        
        assert message.role == "assistant"
        assert message.content == "Default message"
        assert message.provider is None
        assert message.confidence is None
        assert message.metadata == {}
        assert message.tool_calls is None
    
    def test_chat_message_validation(self):
        """Test ChatMessage validation."""
        # Test invalid role
        with pytest.raises(ValueError, match="Invalid role"):
            ChatMessage(role="invalid", content="test")
        
        # Test empty content
        with pytest.raises(ValueError, match="Content cannot be empty"):
            ChatMessage(role="user", content="")
        
        # Test invalid confidence
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            ChatMessage(role="assistant", content="test", confidence=1.5)
    
    def test_chat_message_serialization(self):
        """Test ChatMessage serialization."""
        message = ChatMessage(
            role="user",
            content="Serialize me",
            provider="claude",
            confidence=0.88
        )
        
        serialized = message.to_dict()
        
        assert serialized["role"] == "user"
        assert serialized["content"] == "Serialize me"
        assert serialized["provider"] == "claude"
        assert serialized["confidence"] == 0.88
        assert "timestamp" in serialized
        assert "id" in serialized
    
    def test_chat_message_deserialization(self):
        """Test ChatMessage deserialization."""
        data = {
            "id": "msg-123",
            "role": "assistant",
            "content": "Deserialized message",
            "provider": "openai",
            "confidence": 0.92,
            "timestamp": "2024-01-01T12:00:00",
            "metadata": {"test": True},
            "tool_calls": [{"name": "test_tool"}]
        }
        
        message = ChatMessage.from_dict(data)
        
        assert message.id == "msg-123"
        assert message.role == "assistant"
        assert message.content == "Deserialized message"
        assert message.provider == "openai"
        assert message.confidence == 0.92
        assert message.metadata["test"]
        assert len(message.tool_calls) == 1
    
    def test_message_token_counting(self):
        """Test token counting functionality."""
        message = ChatMessage(
            role="user",
            content="This is a test message with multiple words"
        )
        
        # Mock token counting (simple word-based)
        token_count = message.estimate_tokens()
        assert token_count > 0
        assert token_count >= len(message.content.split())


class TestChatSession:
    """Test ChatSession class."""
    
    def test_chat_session_creation(self):
        """Test ChatSession creation."""
        session = ChatSession(
            context_window=4096,
            max_tokens=2048,
            enable_memory=True,
            metadata={"user": "test_user", "project": "eidolon"}
        )
        
        assert isinstance(session.id, str)
        assert session.context_window == 4096
        assert session.max_tokens == 2048
        assert session.enable_memory
        assert session.metadata["user"] == "test_user"
        assert len(session.messages) == 0
        assert isinstance(session.created_at, datetime)
    
    def test_add_message(self):
        """Test adding messages to session."""
        session = ChatSession()
        
        message1 = ChatMessage("user", "First message")
        message2 = ChatMessage("assistant", "First response")
        
        session.add_message(message1)
        session.add_message(message2)
        
        assert len(session.messages) == 2
        assert session.messages[0] == message1
        assert session.messages[1] == message2
        assert session.updated_at > session.created_at
    
    def test_context_window_management(self):
        """Test context window token management."""
        session = ChatSession(context_window=100)  # Small window for testing
        
        # Add messages until context window is exceeded
        for i in range(10):
            long_message = ChatMessage(
                "user" if i % 2 == 0 else "assistant",
                f"This is a long message number {i} " * 20  # Make it long
            )
            session.add_message(long_message)
        
        # Get active context within window
        active_context = session.get_active_context()
        
        # Should have fewer messages than total
        assert len(active_context["messages"]) < len(session.messages)
        assert active_context["token_count"] <= session.context_window
    
    def test_conversation_summary(self):
        """Test conversation summary generation."""
        session = ChatSession()
        
        session.add_message(ChatMessage("user", "What is Python?"))
        session.add_message(ChatMessage("assistant", "Python is a programming language"))
        session.add_message(ChatMessage("user", "Can you show me an example?"))
        session.add_message(ChatMessage("assistant", "Here's a simple example: print('Hello')"))
        
        summary = session.generate_summary()
        
        assert "Python" in summary
        assert "programming" in summary or "example" in summary
        assert len(summary) < 200  # Should be a summary, not full text
    
    def test_session_serialization(self):
        """Test session serialization."""
        session = ChatSession(metadata={"test": True})
        session.add_message(ChatMessage("user", "Test message"))
        
        serialized = session.to_dict()
        
        assert serialized["id"] == session.id
        assert serialized["context_window"] == session.context_window
        assert serialized["metadata"]["test"]
        assert len(serialized["messages"]) == 1
        assert "created_at" in serialized
    
    def test_session_export_import(self):
        """Test session export and import."""
        original_session = ChatSession()
        original_session.add_message(ChatMessage("user", "Export test"))
        original_session.add_message(ChatMessage("assistant", "Import test"))
        
        # Export
        exported_data = original_session.export(format="json")
        
        # Import
        imported_session = ChatSession.import_from_data(exported_data)
        
        assert imported_session.id != original_session.id  # New ID
        assert len(imported_session.messages) == 2
        assert imported_session.messages[0].content == "Export test"
        assert imported_session.messages[1].content == "Import test"


class TestContextManager:
    """Test ContextManager class."""
    
    @pytest.fixture
    def context_manager(self, mock_memory_system):
        """Create context manager for testing."""
        return ContextManager(memory=mock_memory_system)
    
    async def test_context_retrieval(self, context_manager, mock_memory_system):
        """Test context retrieval from memory."""
        # Setup mock memory response
        mock_context = {
            "recent_screenshots": [
                {"id": "screen-1", "content": "VS Code with Python code"},
                {"id": "screen-2", "content": "Terminal with test output"}
            ],
            "extracted_text": "def test_function():\n    assert True",
            "applications": ["VS Code", "Terminal"],
            "activities": ["coding", "testing"]
        }
        mock_memory_system.get_recent_context.return_value = mock_context
        
        # Get context
        context = await context_manager.get_context(
            time_window=timedelta(minutes=30),
            include_screenshots=True,
            include_text=True
        )
        
        # Verify context
        assert len(context["recent_screenshots"]) == 2
        assert "VS Code" in context["applications"]
        assert "coding" in context["activities"]
        assert "def test_function" in context["extracted_text"]
        
        mock_memory_system.get_recent_context.assert_called_once()
    
    async def test_context_filtering(self, context_manager, mock_memory_system):
        """Test context filtering and relevance scoring."""
        # Setup mock with various content
        mock_memory_system.get_recent_context.return_value = {
            "activities": ["coding", "browsing", "email", "testing"],
            "applications": ["VS Code", "Chrome", "Outlook", "Terminal"],
            "extracted_text": "Mixed content with code and emails"
        }
        
        # Get filtered context for coding
        context = await context_manager.get_relevant_context(
            query="Python programming help",
            relevance_threshold=0.7
        )
        
        # Should prioritize coding-related content
        assert "coding" in context["activities"]
        assert "VS Code" in context["applications"]
        # Email content should be filtered out
        assert "email" not in context.get("activities", [])
    
    async def test_context_summarization(self, context_manager):
        """Test context summarization for large contexts."""
        large_context = {
            "recent_screenshots": [{"content": f"Screenshot {i}"} for i in range(50)],
            "extracted_text": "Very long text content " * 1000,
            "activities": ["coding"] * 100
        }
        
        summarized = await context_manager.summarize_context(
            context=large_context,
            max_length=500
        )
        
        # Should be significantly smaller
        assert len(str(summarized)) < len(str(large_context))
        assert "screenshots" in summarized  # Should maintain key information
        assert "coding" in str(summarized)
    
    async def test_context_caching(self, context_manager, mock_memory_system):
        """Test context caching for performance."""
        mock_memory_system.get_recent_context.return_value = {"test": "data"}
        
        # First call should hit memory
        context1 = await context_manager.get_context(timedelta(minutes=30))
        
        # Second call with same params should use cache
        context2 = await context_manager.get_context(timedelta(minutes=30))
        
        # Should be same data
        assert context1 == context2
        
        # Memory should only be called once (cached)
        assert mock_memory_system.get_recent_context.call_count == 1


@pytest.mark.asyncio
class TestEnhancedInterface:
    """Test EnhancedInterface class."""
    
    @pytest.fixture
    async def enhanced_interface(self, mock_config, mock_memory_system, mock_cloud_api_manager):
        """Create enhanced interface for testing."""
        interface = EnhancedInterface(
            config=mock_config,
            memory=mock_memory_system,
            cloud_api=mock_cloud_api_manager
        )
        yield interface
        await interface.cleanup()
    
    async def test_interface_initialization(self, enhanced_interface, mock_config):
        """Test interface initialization."""
        assert enhanced_interface.config == mock_config
        assert enhanced_interface.memory is not None
        assert enhanced_interface.cloud_api is not None
        assert isinstance(enhanced_interface.sessions, dict)
        assert isinstance(enhanced_interface.context_manager, ContextManager)
    
    async def test_create_chat_session(self, enhanced_interface):
        """Test chat session creation."""
        session = await enhanced_interface.create_chat_session(
            context_window=2048,
            enable_memory=True,
            metadata={"test": True}
        )
        
        assert session is not None
        assert session.context_window == 2048
        assert session.enable_memory
        assert session.metadata["test"]
        assert session.id in enhanced_interface.sessions
    
    async def test_send_message_basic(self, enhanced_interface, mock_cloud_api_manager):
        """Test basic message sending."""
        # Setup mock response
        mock_response = Mock(
            content="Hello! How can I help you?",
            provider="gemini",
            confidence=0.92,
            usage={"tokens": 50}
        )
        mock_cloud_api_manager.analyze_with_context.return_value = mock_response
        
        # Create session and send message
        session = await enhanced_interface.create_chat_session()
        response = await enhanced_interface.send_message(
            session_id=session.id,
            message="Hello, assistant!",
            provider="gemini"
        )
        
        # Verify response
        assert response.content == "Hello! How can I help you?"
        assert response.provider == "gemini"
        assert response.confidence == 0.92
        
        # Verify session has messages
        assert len(session.messages) == 2  # User + assistant
        assert session.messages[0].role == "user"
        assert session.messages[1].role == "assistant"
    
    async def test_send_message_with_context(self, enhanced_interface, mock_cloud_api_manager, mock_memory_system):
        """Test sending message with context."""
        # Setup mocks
        mock_context = {
            "recent_screenshots": [{"content": "User coding Python"}],
            "activities": ["coding"],
            "extracted_text": "import numpy as np"
        }
        mock_memory_system.get_recent_context.return_value = mock_context
        
        mock_response = Mock(
            content="I see you're working with NumPy. How can I help?",
            provider="claude",
            confidence=0.89
        )
        mock_cloud_api_manager.analyze_with_context.return_value = mock_response
        
        # Send message with context
        session = await enhanced_interface.create_chat_session()
        response = await enhanced_interface.send_message(
            session_id=session.id,
            message="What am I working on?",
            include_context=True,
            context_window_minutes=30
        )
        
        # Verify context was used
        call_args = mock_cloud_api_manager.analyze_with_context.call_args
        assert call_args[1]["context"] is not None
        assert "coding" in str(call_args[1]["context"])
        
        # Verify response mentions context
        assert "NumPy" in response.content
    
    async def test_provider_fallback(self, enhanced_interface, mock_cloud_api_manager):
        """Test provider fallback on errors."""
        # Setup primary provider to fail, secondary to succeed
        mock_cloud_api_manager.analyze_with_context.side_effect = [
            Exception("Gemini API error"),
            Mock(
                content="Fallback response from Claude",
                provider="claude",
                confidence=0.85
            )
        ]
        
        session = await enhanced_interface.create_chat_session()
        response = await enhanced_interface.send_message(
            session_id=session.id,
            message="Test fallback",
            provider="gemini",
            fallback_providers=["claude", "openai"]
        )
        
        # Should have fallen back to Claude
        assert response.provider == "claude"
        assert "Fallback response" in response.content
        assert mock_cloud_api_manager.analyze_with_context.call_count == 2
    
    async def test_conversation_history_management(self, enhanced_interface):
        """Test conversation history management."""
        session = await enhanced_interface.create_chat_session()
        
        # Add multiple messages
        for i in range(5):
            await enhanced_interface.send_message(
                session_id=session.id,
                message=f"Message {i}",
                mock_response=True  # Skip actual API call
            )
        
        # Get conversation history
        history = await enhanced_interface.get_conversation_history(session.id)
        
        assert len(history.messages) == 10  # 5 user + 5 assistant
        assert history.session_id == session.id
        assert isinstance(history.created_at, datetime)
        
        # Test history filtering
        user_messages = history.get_messages_by_role("user")
        assert len(user_messages) == 5
        
        recent_messages = history.get_recent_messages(count=4)
        assert len(recent_messages) == 4
    
    async def test_session_persistence(self, enhanced_interface, tmp_path):
        """Test session persistence to disk."""
        # Configure persistence
        enhanced_interface.configure_persistence(
            storage_path=str(tmp_path),
            auto_save=True,
            save_interval=1  # 1 second for testing
        )
        
        # Create session and add messages
        session = await enhanced_interface.create_chat_session()
        await enhanced_interface.send_message(
            session_id=session.id,
            message="Persistent message",
            mock_response=True
        )
        
        # Force save
        await enhanced_interface.save_sessions()
        
        # Verify file exists
        session_files = list(tmp_path.glob("*.json"))
        assert len(session_files) > 0
        
        # Load sessions in new instance
        new_interface = EnhancedInterface(
            config=enhanced_interface.config,
            memory=enhanced_interface.memory,
            cloud_api=enhanced_interface.cloud_api
        )
        new_interface.configure_persistence(storage_path=str(tmp_path))
        await new_interface.load_sessions()
        
        # Verify session was loaded
        loaded_session = await new_interface.get_chat_session(session.id)
        assert loaded_session is not None
        assert len(loaded_session.messages) == 2
    
    async def test_rate_limiting(self, enhanced_interface):
        """Test rate limiting functionality."""
        # Configure rate limiting
        enhanced_interface.configure_rate_limit(
            messages_per_minute=3,
            tokens_per_minute=1000
        )
        
        session = await enhanced_interface.create_chat_session()
        
        # Send messages up to limit
        for i in range(3):
            await enhanced_interface.send_message(
                session_id=session.id,
                message=f"Rate limit test {i}",
                mock_response=True
            )
        
        # Fourth message should be rate limited
        with pytest.raises(RateLimitError):
            await enhanced_interface.send_message(
                session_id=session.id,
                message="This should be rate limited",
                mock_response=True
            )
    
    async def test_streaming_responses(self, enhanced_interface, mock_cloud_api_manager):
        """Test streaming message responses."""
        # Setup streaming mock
        async def mock_stream():
            chunks = ["Hello", " there", "! How", " can I", " help?"]
            for chunk in chunks:
                yield chunk
                await asyncio.sleep(0.01)
        
        mock_cloud_api_manager.stream_with_context = mock_stream
        
        session = await enhanced_interface.create_chat_session()
        
        # Collect streamed response
        chunks = []
        async for chunk in enhanced_interface.send_message_stream(
            session_id=session.id,
            message="Stream test"
        ):
            chunks.append(chunk)
        
        # Verify streaming
        assert len(chunks) == 5
        assert "".join(chunks) == "Hello there! How can I help?"
    
    async def test_multi_modal_support(self, enhanced_interface, mock_cloud_api_manager):
        """Test multi-modal message support."""
        mock_response = Mock(
            content="I can see the image shows a code editor",
            provider="gemini",
            confidence=0.94
        )
        mock_cloud_api_manager.analyze_with_context.return_value = mock_response
        
        session = await enhanced_interface.create_chat_session()
        response = await enhanced_interface.send_message(
            session_id=session.id,
            message="What's in this image?",
            attachments=[{
                "type": "image",
                "data": "base64_image_data",
                "mime_type": "image/png"
            }]
        )
        
        # Verify multi-modal handling
        call_args = mock_cloud_api_manager.analyze_with_context.call_args
        assert "attachments" in call_args[1]
        assert call_args[1]["attachments"][0]["type"] == "image"
        
        assert "image" in response.content
    
    async def test_tool_integration(self, enhanced_interface, mock_cloud_api_manager):
        """Test tool/function calling integration."""
        # Define tools
        tools = [
            {
                "name": "search_memory",
                "description": "Search stored memories",
                "parameters": {"query": {"type": "string"}}
            }
        ]
        
        # Mock tool call response
        mock_response = Mock(
            content="I'll search for that information",
            provider="openai",
            confidence=0.91,
            tool_calls=[{
                "name": "search_memory",
                "arguments": {"query": "Python testing"}
            }]
        )
        mock_cloud_api_manager.analyze_with_context.return_value = mock_response
        
        session = await enhanced_interface.create_chat_session(
            enable_tools=True,
            available_tools=tools
        )
        
        response = await enhanced_interface.send_message(
            session_id=session.id,
            message="Find my previous work on Python testing"
        )
        
        # Verify tool was called
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "search_memory"
    
    async def test_conversation_export_import(self, enhanced_interface):
        """Test conversation export and import."""
        session = await enhanced_interface.create_chat_session()
        
        # Add messages
        await enhanced_interface.send_message(
            session_id=session.id,
            message="First message",
            mock_response=True
        )
        await enhanced_interface.send_message(
            session_id=session.id,
            message="Second message", 
            mock_response=True
        )
        
        # Export conversation
        export_data = await enhanced_interface.export_conversation(
            session_id=session.id,
            format="json"
        )
        
        # Verify export
        assert export_data["session_id"] == session.id
        assert len(export_data["messages"]) == 4
        assert "metadata" in export_data
        
        # Import conversation
        imported_session = await enhanced_interface.import_conversation(
            export_data,
            create_new_session=True
        )
        
        # Verify import
        assert imported_session.id != session.id
        assert len(imported_session.messages) == 4
        assert imported_session.messages[0].content == "First message"
    
    async def test_configuration_management(self, enhanced_interface):
        """Test chat configuration management."""
        # Test default configuration
        config = enhanced_interface.get_chat_config()
        assert config.max_context_window > 0
        assert config.default_provider in ["gemini", "claude", "openai"]
        
        # Update configuration
        new_config = ChatConfiguration(
            max_context_window=8192,
            default_provider="claude",
            enable_streaming=True,
            auto_save=True
        )
        
        enhanced_interface.update_chat_config(new_config)
        
        # Verify configuration updated
        updated_config = enhanced_interface.get_chat_config()
        assert updated_config.max_context_window == 8192
        assert updated_config.default_provider == "claude"
        assert updated_config.enable_streaming
    
    async def test_error_recovery(self, enhanced_interface, mock_cloud_api_manager):
        """Test error recovery mechanisms."""
        # Setup intermittent failures
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Temporary API error")
            return Mock(content="Recovery successful", provider="gemini")
        
        mock_cloud_api_manager.analyze_with_context.side_effect = side_effect
        
        session = await enhanced_interface.create_chat_session()
        
        # Should retry and eventually succeed
        response = await enhanced_interface.send_message(
            session_id=session.id,
            message="Test recovery",
            max_retries=3,
            retry_delay=0.1
        )
        
        assert response.content == "Recovery successful"
        assert call_count == 3  # Failed twice, succeeded on third try


# Mock classes for testing
class ChatMessage:
    """Enhanced ChatMessage with validation."""
    def __init__(self, role: str, content: str, provider: str = None, 
                 confidence: float = None, metadata: Dict = None, 
                 tool_calls: List = None):
        if role not in ["user", "assistant", "system"]:
            raise ValueError(f"Invalid role: {role}")
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")
        if confidence is not None and (confidence < 0 or confidence > 1):
            raise ValueError("Confidence must be between 0 and 1")
        
        self.id = f"msg-{datetime.now().timestamp()}"
        self.role = role
        self.content = content
        self.provider = provider
        self.confidence = confidence
        self.metadata = metadata or {}
        self.tool_calls = tool_calls
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "provider": self.provider,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "tool_calls": self.tool_calls,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ChatMessage':
        """Deserialize from dictionary."""
        msg = cls(
            role=data["role"],
            content=data["content"],
            provider=data.get("provider"),
            confidence=data.get("confidence"),
            metadata=data.get("metadata", {}),
            tool_calls=data.get("tool_calls")
        )
        msg.id = data["id"]
        return msg
    
    def estimate_tokens(self) -> int:
        """Estimate token count (simple implementation)."""
        return len(self.content.split()) + len(str(self.metadata))


class ChatSession:
    """Enhanced ChatSession with context management."""
    def __init__(self, context_window: int = 4096, max_tokens: int = 2048,
                 enable_memory: bool = False, metadata: Dict = None):
        self.id = f"session-{datetime.now().timestamp()}"
        self.context_window = context_window
        self.max_tokens = max_tokens
        self.enable_memory = enable_memory
        self.metadata = metadata or {}
        self.messages: List[ChatMessage] = []
        self.created_at = datetime.now()
        self.updated_at = self.created_at
    
    def add_message(self, message: ChatMessage):
        """Add message to session."""
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def get_active_context(self) -> Dict:
        """Get active context within token window."""
        total_tokens = 0
        active_messages = []
        
        # Work backwards from most recent
        for message in reversed(self.messages):
            msg_tokens = message.estimate_tokens()
            if total_tokens + msg_tokens > self.context_window:
                break
            
            active_messages.insert(0, message)
            total_tokens += msg_tokens
        
        return {
            "messages": active_messages,
            "token_count": total_tokens
        }
    
    def generate_summary(self) -> str:
        """Generate conversation summary."""
        if not self.messages:
            return "Empty conversation"
        
        # Simple summary based on content
        content = " ".join(msg.content for msg in self.messages[:5])
        words = content.split()
        return " ".join(words[:30]) + "..." if len(words) > 30 else content
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "context_window": self.context_window,
            "max_tokens": self.max_tokens,
            "enable_memory": self.enable_memory,
            "metadata": self.metadata,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    def export(self, format: str = "json") -> Dict:
        """Export session data."""
        return {
            "format": format,
            "session": self.to_dict(),
            "export_time": datetime.now().isoformat()
        }
    
    @classmethod
    def import_from_data(cls, data: Dict) -> 'ChatSession':
        """Import session from data."""
        session_data = data.get("session", data)
        session = cls(
            context_window=session_data.get("context_window", 4096),
            max_tokens=session_data.get("max_tokens", 2048),
            enable_memory=session_data.get("enable_memory", False),
            metadata=session_data.get("metadata", {})
        )
        
        # Import messages
        for msg_data in session_data.get("messages", []):
            message = ChatMessage.from_dict(msg_data)
            session.add_message(message)
        
        return session


class ConversationHistory:
    """Conversation history management."""
    def __init__(self, session_id: str, messages: List[ChatMessage]):
        self.session_id = session_id
        self.messages = messages
        self.created_at = datetime.now()
    
    def get_messages_by_role(self, role: str) -> List[ChatMessage]:
        """Get messages by role."""
        return [msg for msg in self.messages if msg.role == role]
    
    def get_recent_messages(self, count: int) -> List[ChatMessage]:
        """Get recent messages."""
        return self.messages[-count:] if count < len(self.messages) else self.messages


class ContextManager:
    """Context management for chat."""
    def __init__(self, memory):
        self.memory = memory
        self._cache = {}
        self._cache_timeout = 300  # 5 minutes
    
    async def get_context(self, time_window: timedelta, **kwargs) -> Dict:
        """Get context from memory."""
        cache_key = f"{time_window.total_seconds()}_{hash(str(kwargs))}"
        
        # Check cache
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if datetime.now() - cached_time < timedelta(seconds=self._cache_timeout):
                return cached_data
        
        # Get from memory
        context = await self.memory.get_recent_context(time_window=time_window)
        
        # Cache result
        self._cache[cache_key] = (datetime.now(), context)
        
        return context
    
    async def get_relevant_context(self, query: str, relevance_threshold: float = 0.7) -> Dict:
        """Get relevant context for query."""
        context = await self.memory.get_recent_context(time_window=timedelta(hours=1))
        
        # Simple relevance filtering (in practice would use embeddings)
        query_words = set(query.lower().split())
        
        filtered_activities = []
        for activity in context.get("activities", []):
            if any(word in activity.lower() for word in query_words):
                filtered_activities.append(activity)
        
        filtered_apps = []
        for app in context.get("applications", []):
            if any(word in app.lower() for word in query_words):
                filtered_apps.append(app)
        
        return {
            "activities": filtered_activities,
            "applications": filtered_apps,
            "extracted_text": context.get("extracted_text", "")
        }
    
    async def summarize_context(self, context: Dict, max_length: int = 500) -> Dict:
        """Summarize large context."""
        # Simple summarization
        summarized = {}
        
        for key, value in context.items():
            if isinstance(value, list) and len(value) > 10:
                summarized[key] = value[:5] + [f"... and {len(value) - 5} more"]
            elif isinstance(value, str) and len(value) > max_length:
                summarized[key] = value[:max_length] + "..."
            else:
                summarized[key] = value
        
        return summarized


class ChatConfiguration:
    """Chat configuration class."""
    def __init__(self, max_context_window: int = 4096, default_provider: str = "gemini",
                 enable_streaming: bool = True, auto_save: bool = True):
        self.max_context_window = max_context_window
        self.default_provider = default_provider
        self.enable_streaming = enable_streaming
        self.auto_save = auto_save


class RateLimitError(Exception):
    """Rate limit exceeded error."""
    pass


class EnhancedInterface:
    """Enhanced interface with chat capabilities."""
    def __init__(self, config, memory, cloud_api):
        self.config = config
        self.memory = memory
        self.cloud_api = cloud_api
        self.sessions: Dict[str, ChatSession] = {}
        self.context_manager = ContextManager(memory)
        self._chat_config = ChatConfiguration()
        self._rate_limiter = None
        self._persistence_config = None
    
    async def create_chat_session(self, **kwargs) -> ChatSession:
        """Create new chat session."""
        session = ChatSession(**kwargs)
        self.sessions[session.id] = session
        return session
    
    async def send_message(self, session_id: str, message: str, **kwargs) -> ChatMessage:
        """Send message and get response."""
        session = self.sessions[session_id]
        
        # Rate limiting check
        if self._rate_limiter:
            self._check_rate_limit()
        
        # Add user message
        user_msg = ChatMessage("user", message)
        session.add_message(user_msg)
        
        # Mock response for testing
        if kwargs.get("mock_response"):
            response_msg = ChatMessage(
                "assistant",
                f"Mock response to: {message}",
                provider="mock"
            )
            session.add_message(response_msg)
            return response_msg
        
        # Get context if requested
        context = None
        if kwargs.get("include_context"):
            window_minutes = kwargs.get("context_window_minutes", 30)
            context = await self.context_manager.get_context(
                time_window=timedelta(minutes=window_minutes)
            )
        
        # Prepare API call
        api_kwargs = {
            "prompt": message,
            "context": context,
            "provider": kwargs.get("provider", self._chat_config.default_provider)
        }
        
        if kwargs.get("attachments"):
            api_kwargs["attachments"] = kwargs["attachments"]
        
        # Handle provider fallback
        providers = [api_kwargs["provider"]] + kwargs.get("fallback_providers", [])
        
        for provider in providers:
            try:
                api_kwargs["provider"] = provider
                response = await self.cloud_api.analyze_with_context(**api_kwargs)
                
                # Create assistant message
                assistant_msg = ChatMessage(
                    "assistant",
                    response.content,
                    provider=response.provider,
                    confidence=response.confidence,
                    tool_calls=getattr(response, "tool_calls", None)
                )
                session.add_message(assistant_msg)
                return assistant_msg
                
            except Exception as e:
                if provider == providers[-1]:  # Last provider
                    raise e
                continue  # Try next provider
    
    async def send_message_stream(self, session_id: str, message: str):
        """Stream message response."""
        async for chunk in self.cloud_api.stream_with_context(message):
            yield chunk
    
    async def get_chat_session(self, session_id: str) -> Optional[ChatSession]:
        """Get chat session."""
        return self.sessions.get(session_id)
    
    async def get_conversation_history(self, session_id: str) -> ConversationHistory:
        """Get conversation history."""
        session = self.sessions[session_id]
        return ConversationHistory(session_id, session.messages)
    
    async def export_conversation(self, session_id: str, format: str = "json") -> Dict:
        """Export conversation."""
        session = self.sessions[session_id]
        return session.export(format)
    
    async def import_conversation(self, data: Dict, create_new_session: bool = True) -> ChatSession:
        """Import conversation."""
        session = ChatSession.import_from_data(data)
        if create_new_session or session.id not in self.sessions:
            self.sessions[session.id] = session
        return session
    
    def configure_rate_limit(self, messages_per_minute: int, tokens_per_minute: int):
        """Configure rate limiting."""
        self._rate_limiter = {
            "messages_per_minute": messages_per_minute,
            "tokens_per_minute": tokens_per_minute,
            "message_count": 0,
            "last_reset": datetime.now()
        }
    
    def _check_rate_limit(self):
        """Check rate limit."""
        if not self._rate_limiter:
            return
        
        now = datetime.now()
        if (now - self._rate_limiter["last_reset"]).total_seconds() >= 60:
            self._rate_limiter["message_count"] = 0
            self._rate_limiter["last_reset"] = now
        
        if self._rate_limiter["message_count"] >= self._rate_limiter["messages_per_minute"]:
            raise RateLimitError("Message rate limit exceeded")
        
        self._rate_limiter["message_count"] += 1
    
    def configure_persistence(self, storage_path: str, auto_save: bool = True, save_interval: int = 300):
        """Configure session persistence."""
        self._persistence_config = {
            "storage_path": storage_path,
            "auto_save": auto_save,
            "save_interval": save_interval
        }
    
    async def save_sessions(self):
        """Save sessions to disk."""
        if not self._persistence_config:
            return
        
        # Mock save implementation
        import json
        from pathlib import Path
        
        storage_path = Path(self._persistence_config["storage_path"])
        storage_path.mkdir(exist_ok=True)
        
        for session in self.sessions.values():
            session_file = storage_path / f"{session.id}.json"
            with open(session_file, "w") as f:
                json.dump(session.to_dict(), f)
    
    async def load_sessions(self):
        """Load sessions from disk."""
        if not self._persistence_config:
            return
        
        # Mock load implementation
        import json
        from pathlib import Path
        
        storage_path = Path(self._persistence_config["storage_path"])
        if not storage_path.exists():
            return
        
        for session_file in storage_path.glob("*.json"):
            with open(session_file, "r") as f:
                session_data = json.load(f)
                session = ChatSession.import_from_data({"session": session_data})
                self.sessions[session.id] = session
    
    def get_chat_config(self) -> ChatConfiguration:
        """Get chat configuration."""
        return self._chat_config
    
    def update_chat_config(self, config: ChatConfiguration):
        """Update chat configuration."""
        self._chat_config = config
    
    async def cleanup(self):
        """Cleanup resources."""
        if self._persistence_config and self._persistence_config["auto_save"]:
            await self.save_sessions()
        self.sessions.clear()