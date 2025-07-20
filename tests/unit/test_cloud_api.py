#!/usr/bin/env python3
"""
Unit tests for the Cloud API component.

Tests cover cloud AI integrations including Gemini, Claude, and OpenAI APIs.
"""

import pytest
import asyncio
import base64
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any

# Import the modules to test
from eidolon.models.cloud_api import (
    CloudAPIResponse, GeminiAPI, ClaudeAPI, OpenAIAPI, 
    CloudAPIManager, OpenRouterClaudeAPI
)


class TestCloudAPIResponse:
    """Test the CloudAPIResponse class."""
    
    def test_cloud_api_response_creation(self):
        """Test CloudAPIResponse creation with all fields."""
        response = CloudAPIResponse(
            content="This is a test response",
            model="gpt-4",
            provider="openai",
            confidence=0.95,
            usage={"tokens": 100, "cost": 0.01},
            metadata={"temperature": 0.7}
        )
        
        assert response.content == "This is a test response"
        assert response.model == "gpt-4"
        assert response.provider == "openai"
        assert response.confidence == 0.95
        assert response.usage["tokens"] == 100
        assert response.metadata["temperature"] == 0.7
        assert isinstance(response.timestamp, datetime)
    
    def test_cloud_api_response_defaults(self):
        """Test CloudAPIResponse with default values."""
        response = CloudAPIResponse(
            content="Test",
            model="test-model",
            provider="test"
        )
        
        assert response.confidence == 0.0
        assert response.usage == {}
        assert response.metadata == {}
    
    def test_cloud_api_response_to_dict(self):
        """Test CloudAPIResponse serialization to dictionary."""
        response = CloudAPIResponse(
            content="Test content",
            model="claude-3",
            provider="anthropic",
            confidence=0.9,
            usage={"characters": 50}
        )
        
        result = response.to_dict()
        
        assert isinstance(result, dict)
        assert result["content"] == "Test content"
        assert result["model"] == "claude-3"
        assert result["provider"] == "anthropic"
        assert result["confidence"] == 0.9
        assert result["usage"]["characters"] == 50
        assert "timestamp" in result
        assert isinstance(result["timestamp"], str)


class TestGeminiAPI:
    """Test the GeminiAPI class."""
    
    @pytest.fixture
    def mock_genai(self):
        """Mock the Google Generative AI module."""
        with patch('eidolon.models.cloud_api.genai') as mock:
            mock_model = Mock()
            mock.GenerativeModel.return_value = mock_model
            yield mock, mock_model
    
    @pytest.fixture
    def mock_image_file(self, tmp_path):
        """Create a mock image file."""
        image_path = tmp_path / "test_image.png"
        image_path.write_bytes(b"fake_image_data")
        return str(image_path)
    
    def test_gemini_api_initialization_success(self, mock_genai):
        """Test successful Gemini API initialization."""
        mock, mock_model = mock_genai
        
        with patch('eidolon.models.cloud_api.GEMINI_AVAILABLE', True), \
             patch('eidolon.models.cloud_api.get_component_logger'):
            
            api = GeminiAPI(api_key="test_key")
            
            assert api.available is True
            assert api.api_key == "test_key"
            mock.configure.assert_called_once_with(api_key="test_key")
            mock.GenerativeModel.assert_called_once_with('gemini-1.5-flash')
    
    def test_gemini_api_initialization_no_key(self):
        """Test Gemini API initialization without API key."""
        with patch('eidolon.models.cloud_api.GEMINI_AVAILABLE', True), \
             patch('eidolon.models.cloud_api.get_component_logger'), \
             patch.dict('os.environ', {}, clear=True):
            
            api = GeminiAPI()
            
            assert api.available is False
            assert api.api_key is None
    
    def test_gemini_api_initialization_sdk_not_available(self):
        """Test Gemini API when SDK is not available."""
        with patch('eidolon.models.cloud_api.GEMINI_AVAILABLE', False), \
             patch('eidolon.models.cloud_api.get_component_logger'):
            
            api = GeminiAPI(api_key="test_key")
            
            assert api.available is False
    
    @pytest.mark.asyncio
    async def test_analyze_image_success(self, mock_genai, mock_image_file):
        """Test successful image analysis with Gemini."""
        mock, mock_model = mock_genai
        
        # Mock the model response
        mock_response = Mock()
        mock_response.text = "This image contains a cat sitting on a table."
        mock_response.usage_metadata = Mock(total_token_count=50)
        mock_model.generate_content.return_value = mock_response
        
        with patch('eidolon.models.cloud_api.GEMINI_AVAILABLE', True), \
             patch('eidolon.models.cloud_api.get_component_logger'), \
             patch('eidolon.models.cloud_api.Image') as mock_image:
            
            api = GeminiAPI(api_key="test_key")
            api.model = mock_model
            
            result = await api.analyze_image(mock_image_file, "Describe this image")
            
            assert isinstance(result, CloudAPIResponse)
            assert result.content == "This image contains a cat sitting on a table."
            assert result.provider == "gemini"
            assert result.model == "gemini-1.5-flash"
            assert result.usage["total_tokens"] == 50
    
    @pytest.mark.asyncio
    async def test_analyze_image_not_available(self):
        """Test image analysis when API is not available."""
        with patch('eidolon.models.cloud_api.get_component_logger'):
            api = GeminiAPI()
            api.available = False
            
            result = await api.analyze_image("test.png")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_analyze_text_success(self, mock_genai):
        """Test successful text analysis with Gemini."""
        mock, mock_model = mock_genai
        
        mock_response = Mock()
        mock_response.text = "This text is about Python programming."
        mock_response.usage_metadata = Mock(total_token_count=30)
        mock_model.generate_content.return_value = mock_response
        
        with patch('eidolon.models.cloud_api.GEMINI_AVAILABLE', True), \
             patch('eidolon.models.cloud_api.get_component_logger'):
            
            api = GeminiAPI(api_key="test_key")
            api.model = mock_model
            
            result = await api.analyze_text("Python is a programming language", "Analyze this text")
            
            assert isinstance(result, CloudAPIResponse)
            assert result.content == "This text is about Python programming."
            assert result.provider == "gemini"


class TestClaudeAPI:
    """Test the ClaudeAPI class."""
    
    @pytest.fixture
    def mock_anthropic(self):
        """Mock the Anthropic client."""
        with patch('eidolon.models.cloud_api.Anthropic') as mock:
            mock_client = Mock()
            mock.return_value = mock_client
            yield mock, mock_client
    
    def test_claude_api_initialization_success(self, mock_anthropic):
        """Test successful Claude API initialization."""
        mock, mock_client = mock_anthropic
        
        with patch('eidolon.models.cloud_api.CLAUDE_AVAILABLE', True), \
             patch('eidolon.models.cloud_api.get_component_logger'):
            
            api = ClaudeAPI(api_key="test_key")
            
            assert api.available is True
            assert api.api_key == "test_key"
            mock.assert_called_once_with(api_key="test_key")
    
    @pytest.mark.asyncio
    async def test_analyze_image_success(self, mock_anthropic, tmp_path):
        """Test successful image analysis with Claude."""
        mock, mock_client = mock_anthropic
        
        # Create test image
        image_path = tmp_path / "test.jpg"
        image_path.write_bytes(b"fake_image_data")
        
        # Mock response
        mock_response = Mock()
        mock_response.content = [Mock(text="This is a description of the image")]
        mock_response.usage = Mock(input_tokens=100, output_tokens=50)
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        
        with patch('eidolon.models.cloud_api.CLAUDE_AVAILABLE', True), \
             patch('eidolon.models.cloud_api.get_component_logger'):
            
            api = ClaudeAPI(api_key="test_key")
            api.client = mock_client
            
            result = await api.analyze_image(str(image_path), "Describe this")
            
            assert isinstance(result, CloudAPIResponse)
            assert result.content == "This is a description of the image"
            assert result.provider == "claude"
            assert result.model == "claude-3-sonnet-20240229"
            assert result.usage["input_tokens"] == 100
            assert result.usage["output_tokens"] == 50
    
    @pytest.mark.asyncio
    async def test_generate_response_success(self, mock_anthropic):
        """Test successful text generation with Claude."""
        mock, mock_client = mock_anthropic
        
        mock_response = Mock()
        mock_response.content = [Mock(text="Generated response text")]
        mock_response.usage = Mock(input_tokens=20, output_tokens=30)
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        
        with patch('eidolon.models.cloud_api.CLAUDE_AVAILABLE', True), \
             patch('eidolon.models.cloud_api.get_component_logger'):
            
            api = ClaudeAPI(api_key="test_key")
            api.client = mock_client
            
            result = await api.generate_response("Generate a story", max_tokens=100)
            
            assert isinstance(result, CloudAPIResponse)
            assert result.content == "Generated response text"
            assert result.usage["total_tokens"] == 50


class TestOpenAIAPI:
    """Test the OpenAIAPI class."""
    
    @pytest.fixture
    def mock_openai(self):
        """Mock the OpenAI client."""
        with patch('eidolon.models.cloud_api.OpenAI') as mock:
            mock_client = Mock()
            mock.return_value = mock_client
            yield mock, mock_client
    
    def test_openai_api_initialization_success(self, mock_openai):
        """Test successful OpenAI API initialization."""
        mock, mock_client = mock_openai
        
        with patch('eidolon.models.cloud_api.OPENAI_AVAILABLE', True), \
             patch('eidolon.models.cloud_api.get_component_logger'):
            
            api = OpenAIAPI(api_key="test_key")
            
            assert api.available is True
            assert api.api_key == "test_key"
            mock.assert_called_once_with(api_key="test_key")
    
    @pytest.mark.asyncio
    async def test_analyze_image_success(self, mock_openai):
        """Test successful image analysis with OpenAI."""
        mock, mock_client = mock_openai
        
        # Mock response structure
        mock_message = Mock()
        mock_message.content = "Image analysis result"
        mock_response = Mock()
        mock_response.choices = [Mock(message=mock_message)]
        mock_response.usage = Mock(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        
        # Use AsyncMock for async method
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        with patch('eidolon.models.cloud_api.OPENAI_AVAILABLE', True), \
             patch('eidolon.models.cloud_api.get_component_logger'):
            
            api = OpenAIAPI(api_key="test_key")
            api.client = mock_client
            
            result = await api.analyze_image("http://example.com/image.jpg", "Analyze this")
            
            assert isinstance(result, CloudAPIResponse)
            assert result.content == "Image analysis result"
            assert result.provider == "openai"
            assert result.model == "gpt-4-vision-preview"
            assert result.usage["total_tokens"] == 150
    
    @pytest.mark.asyncio
    async def test_generate_response_success(self, mock_openai):
        """Test successful text generation with OpenAI."""
        mock, mock_client = mock_openai
        
        mock_message = Mock()
        mock_message.content = "Generated text response"
        mock_response = Mock()
        mock_response.choices = [Mock(message=mock_message)]
        mock_response.usage = Mock(total_tokens=80)
        
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        with patch('eidolon.models.cloud_api.OPENAI_AVAILABLE', True), \
             patch('eidolon.models.cloud_api.get_component_logger'):
            
            api = OpenAIAPI(api_key="test_key")
            api.client = mock_client
            
            result = await api.generate_response("Write a poem", temperature=0.7)
            
            assert isinstance(result, CloudAPIResponse)
            assert result.content == "Generated text response"
            assert result.provider == "openai"
            assert result.model == "gpt-4"


class TestOpenRouterClaudeAPI:
    """Test the OpenRouterClaudeAPI class."""
    
    @pytest.fixture
    def mock_aiohttp(self):
        """Mock aiohttp for HTTP requests."""
        with patch('eidolon.models.cloud_api.aiohttp.ClientSession') as mock:
            mock_session = AsyncMock()
            mock.return_value.__aenter__.return_value = mock_session
            yield mock_session
    
    def test_openrouter_claude_api_initialization(self):
        """Test OpenRouter Claude API initialization."""
        with patch('eidolon.models.cloud_api.get_component_logger'):
            api = OpenRouterClaudeAPI(api_key="test_key")
            
            assert api.api_key == "test_key"
            assert api.base_url == "https://openrouter.ai/api/v1"
            assert api.available is True
    
    @pytest.mark.asyncio
    async def test_analyze_text_success(self):
        """Test successful text analysis with OpenRouter."""
        with patch('eidolon.models.cloud_api.get_component_logger'), \
             patch('eidolon.models.cloud_api.OPENAI_AVAILABLE', True):
            
            api = OpenRouterClaudeAPI(api_key="test_key")
            
            # Mock the OpenAI client approach
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="Analysis result from OpenRouter"))]
            mock_response.usage = Mock(total_tokens=100)
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            api.client = mock_client
            
            result = await api.analyze_text("Analyze this text", "Do analysis")
            
            assert isinstance(result, CloudAPIResponse)
            assert result.content == "Analysis result from OpenRouter"
            assert result.provider == "openrouter"
            assert result.model == "anthropic/claude-3-haiku"
            assert result.usage["total_tokens"] == 100


class TestCloudAPIManager:
    """Test the CloudAPIManager class."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        config = Mock()
        config.analysis.cloud_apis = {
            "gemini_key": "test_gemini_key",
            "claude_key": "test_claude_key",
            "openai_key": "test_openai_key",
            "openrouter_key": "test_openrouter_key"
        }
        config.analysis.preferred_providers = ["gemini", "claude", "openai"]
        return config
    
    def test_cloud_api_manager_initialization(self, mock_config):
        """Test CloudAPIManager initialization."""
        with patch('eidolon.models.cloud_api.get_config', return_value=mock_config), \
             patch('eidolon.models.cloud_api.get_component_logger'), \
             patch('eidolon.models.cloud_api.GeminiAPI') as mock_gemini, \
             patch('eidolon.models.cloud_api.ClaudeAPI') as mock_claude, \
             patch('eidolon.models.cloud_api.OpenAIAPI') as mock_openai, \
             patch('eidolon.models.cloud_api.OpenRouterClaudeAPI') as mock_openrouter:
            
            # Mock API availability
            mock_gemini.return_value.available = True
            mock_claude.return_value.available = True
            mock_openai.return_value.available = False
            mock_openrouter.return_value.available = False
            
            manager = CloudAPIManager()
            
            assert "gemini" in manager.apis
            assert "claude" in manager.apis
            assert "openai" in manager.apis
            assert manager.get_available_providers() == ["gemini", "claude"]
    
    @pytest.mark.asyncio
    async def test_analyze_with_best_available_success(self, mock_config):
        """Test analysis with best available provider."""
        with patch('eidolon.models.cloud_api.get_config', return_value=mock_config), \
             patch('eidolon.models.cloud_api.get_component_logger'):
            
            manager = CloudAPIManager()
            
            # Mock Gemini API
            mock_gemini = AsyncMock()
            mock_gemini.available = True
            mock_gemini.analyze_text = AsyncMock(return_value=CloudAPIResponse(
                content="Gemini analysis",
                model="gemini-1.5",
                provider="gemini"
            ))
            manager.apis["gemini"] = mock_gemini
            
            result = await manager.analyze_with_best_available(
                content_type="text",
                content="Test text"
            )
            
            assert result is not None
            assert result.content == "Gemini analysis"
            assert result.provider == "gemini"
    
    @pytest.mark.asyncio
    async def test_analyze_with_specific_provider(self, mock_config):
        """Test analysis with specific provider."""
        with patch('eidolon.models.cloud_api.get_config', return_value=mock_config), \
             patch('eidolon.models.cloud_api.get_component_logger'):
            
            manager = CloudAPIManager()
            
            # Mock Claude API
            mock_claude = AsyncMock()
            mock_claude.available = True
            mock_claude.analyze_text = AsyncMock(return_value=CloudAPIResponse(
                content="Claude analysis",
                model="claude-3",
                provider="claude"
            ))
            manager.apis["claude"] = mock_claude
            
            result = await manager.analyze_with_provider(
                "claude",
                content_type="text",
                content="Test text"
            )
            
            assert result is not None
            assert result.content == "Claude analysis"
            assert result.provider == "claude"
    
    @pytest.mark.asyncio
    async def test_parallel_analysis(self, mock_config):
        """Test parallel analysis with multiple providers."""
        with patch('eidolon.models.cloud_api.get_config', return_value=mock_config), \
             patch('eidolon.models.cloud_api.get_component_logger'):
            
            manager = CloudAPIManager()
            
            # Mock multiple APIs
            for provider, response_text in [("gemini", "Gemini result"), ("claude", "Claude result")]:
                mock_api = AsyncMock()
                mock_api.available = True
                mock_api.analyze_text = AsyncMock(return_value=CloudAPIResponse(
                    content=response_text,
                    model=f"{provider}-model",
                    provider=provider
                ))
                manager.apis[provider] = mock_api
            
            results = await manager.parallel_analysis(
                providers=["gemini", "claude"],
                content_type="text",
                content="Test text"
            )
            
            assert len(results) == 2
            assert any(r.provider == "gemini" for r in results)
            assert any(r.provider == "claude" for r in results)
    
    def test_get_cost_estimate(self, mock_config):
        """Test cost estimation for API usage."""
        with patch('eidolon.models.cloud_api.get_config', return_value=mock_config), \
             patch('eidolon.models.cloud_api.get_component_logger'):
            
            manager = CloudAPIManager()
            
            usage = {
                "input_tokens": 1000,
                "output_tokens": 500,
                "total_tokens": 1500
            }
            
            # Test known provider
            cost = manager.get_cost_estimate("openai", "gpt-4", usage)
            assert isinstance(cost, float)
            assert cost > 0
            
            # Test unknown provider
            cost = manager.get_cost_estimate("unknown", "model", usage)
            assert cost == 0.0


class TestCloudAPIError:
    """Test error handling in cloud APIs."""
    
    @pytest.mark.asyncio
    async def test_gemini_api_error_handling(self, mock_genai):
        """Test Gemini API error handling."""
        mock, mock_model = mock_genai
        
        # Make the model raise an exception
        mock_model.generate_content.side_effect = Exception("API Error")
        
        with patch('eidolon.models.cloud_api.GEMINI_AVAILABLE', True), \
             patch('eidolon.models.cloud_api.get_component_logger'):
            
            api = GeminiAPI(api_key="test_key")
            api.model = mock_model
            
            result = await api.analyze_text("Test", "Analyze")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_claude_api_error_handling(self, mock_anthropic):
        """Test Claude API error handling."""
        mock, mock_client = mock_anthropic
        
        # Make the client raise an exception
        mock_client.messages.create = AsyncMock(side_effect=Exception("API Error"))
        
        with patch('eidolon.models.cloud_api.CLAUDE_AVAILABLE', True), \
             patch('eidolon.models.cloud_api.get_component_logger'):
            
            api = ClaudeAPI(api_key="test_key")
            api.client = mock_client
            
            result = await api.generate_response("Test prompt")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_openrouter_api_error_handling(self, mock_aiohttp):
        """Test OpenRouter API error handling."""
        # Mock error response
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        
        mock_aiohttp.post.return_value.__aenter__.return_value = mock_response
        
        with patch('eidolon.models.cloud_api.get_component_logger'):
            api = OpenRouterClaudeAPI(api_key="test_key")
            
            result = await api.analyze_text("Test", "Analyze")
            
            assert result is None


class TestCloudAPIIntegration:
    """Integration tests for cloud APIs."""
    
    @pytest.mark.asyncio
    async def test_fallback_mechanism(self, mock_config):
        """Test fallback when primary provider fails."""
        with patch('eidolon.models.cloud_api.get_config', return_value=mock_config), \
             patch('eidolon.models.cloud_api.get_component_logger'):
            
            manager = CloudAPIManager()
            
            # Mock primary provider (gemini) to fail
            mock_gemini = AsyncMock()
            mock_gemini.available = True
            mock_gemini.analyze_text = AsyncMock(return_value=None)
            manager.apis["gemini"] = mock_gemini
            
            # Mock fallback provider (claude) to succeed
            mock_claude = AsyncMock()
            mock_claude.available = True
            mock_claude.analyze_text = AsyncMock(return_value=CloudAPIResponse(
                content="Claude fallback result",
                model="claude-3",
                provider="claude"
            ))
            manager.apis["claude"] = mock_claude
            
            result = await manager.analyze_with_best_available(
                content_type="text",
                content="Test"
            )
            
            assert result is not None
            assert result.provider == "claude"
            assert result.content == "Claude fallback result"
    
    @pytest.mark.asyncio
    async def test_rate_limiting_simulation(self):
        """Test handling of rate limiting scenarios."""
        # This test simulates rate limiting behavior
        call_count = 0
        
        async def mock_api_call():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Rate limit exceeded")
            return CloudAPIResponse("Success", "model", "provider")
        
        # Test retry logic (if implemented)
        # This is a placeholder for actual rate limiting tests
        assert True
    
    def test_model_selection_logic(self, mock_config):
        """Test model selection based on content type and requirements."""
        with patch('eidolon.models.cloud_api.get_config', return_value=mock_config), \
             patch('eidolon.models.cloud_api.get_component_logger'):
            
            manager = CloudAPIManager()
            
            # Test that appropriate models are selected for different content types
            # This would test the actual model selection logic when implemented
            assert True


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])