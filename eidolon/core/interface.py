"""
Interface component for Eidolon AI Personal Assistant

Handles user interactions, natural language queries, and system responses.
Provides the main entry point for user commands and queries.
"""

import asyncio
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import deque

from ..utils.logging import get_component_logger, log_performance
from ..utils.config import get_config
from ..models.cloud_api import CloudAPIManager
from ..core.memory import MemorySystem, SearchResult
from ..storage.metadata_db import MetadataDatabase
from ..core.observer import Observer


class ConversationTurn:
    """Represents a single turn in a conversation."""
    
    def __init__(self, role: str, content: str, timestamp: Optional[datetime] = None):
        self.role = role  # "user" or "assistant"
        self.content = content
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }


class ConversationHistory:
    """Manages conversation history with context window."""
    
    def __init__(self, max_turns: int = 10):
        self.turns: deque[ConversationTurn] = deque(maxlen=max_turns * 2)  # user + assistant
        self.max_turns = max_turns
    
    def add_turn(self, role: str, content: str) -> None:
        """Add a conversation turn."""
        self.turns.append(ConversationTurn(role, content))
    
    def get_context(self, max_tokens: int = 2000) -> List[Dict[str, str]]:
        """Get conversation context for LLM."""
        messages = []
        total_tokens = 0
        
        # Process turns in reverse order (most recent first)
        for turn in reversed(self.turns):
            # Rough token estimation: ~4 characters per token
            turn_tokens = len(turn.content) // 4
            
            if total_tokens + turn_tokens > max_tokens:
                break
            
            messages.insert(0, {"role": turn.role, "content": turn.content})
            total_tokens += turn_tokens
        
        return messages
    
    def clear(self) -> None:
        """Clear conversation history."""
        self.turns.clear()


class QueryResult:
    """Represents the result of processing a user query."""
    
    def __init__(
        self,
        query: str,
        response: str,
        confidence: float,
        sources: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        context_used: Optional[List[SearchResult]] = None
    ):
        self.query = query
        self.response = response
        self.confidence = confidence
        self.sources = sources or []
        self.metadata = metadata or {}
        self.context_used = context_used or []
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "response": self.response,
            "confidence": self.confidence,
            "sources": self.sources,
            "metadata": self.metadata,
            "context_used": [ctx.to_dict() for ctx in self.context_used],
            "timestamp": self.timestamp.isoformat()
        }


class Interface:
    """
    User interface and query processing system for Eidolon.
    
    Provides LLM-powered chat with screen memory context, conversational AI,
    and intelligent query processing.
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_component_logger("interface")
        
        # Initialize components
        self.cloud_api = CloudAPIManager()
        self.memory_system = MemorySystem()
        self.metadata_db = MetadataDatabase()
        self.observer = None  # Lazy load when needed
        
        # Conversation management
        self.conversation_history = ConversationHistory(max_turns=10)
        self.system_prompt = self._build_system_prompt()
        
        # Available providers
        self.available_providers = self.cloud_api.get_available_providers()
        self.preferred_provider = self._select_preferred_provider()
        
        self.logger.info(f"Interface initialized with chat capabilities. Available providers: {self.available_providers}")
    
    def _select_preferred_provider(self) -> Optional[str]:
        """Select the best available provider for chat."""
        # Prioritize Gemini for cost-effectiveness
        provider_priority = ["gemini", "claude", "openrouter", "openai"]
        
        for provider in provider_priority:
            if provider in self.available_providers:
                return provider
        
        return None
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the LLM."""
        return """You are Eidolon, an AI personal assistant with access to the user's screen capture history.

You help users understand and analyze their computer activity by:
- Searching through their captured screenshots and extracted text
- Answering questions about what they were doing at specific times
- Finding specific content they saw on their screen
- Providing insights about their work patterns and activities
- Helping them recall information from their digital activities

When answering questions:
1. Use the provided screen capture context to give accurate, specific answers
2. Reference timestamps and applications when relevant
3. Be conversational and helpful
4. If you can't find relevant information, suggest what might help
5. Respect privacy - only discuss what's in the provided context

Current capabilities:
- Full-text search across all captured content
- Semantic search for finding related content
- Time-based queries (today, yesterday, this week, etc.)
- Application and content-type filtering
- Pattern analysis and insights"""
    
    async def _get_context_for_query(self, query: str) -> Tuple[str, List[SearchResult]]:
        """Get relevant context from screen captures for the query."""
        try:
            # Use enhanced conversation context from memory system
            context_data = await self.memory_system.get_conversation_context(
                query,
                time_window_hours=24,
                max_results=10
            )
            
            # Build comprehensive context string
            context_parts = []
            
            # Add search results
            if context_data["search_results"]:
                context_parts.append("## Relevant Screen Captures:")
                for i, result in enumerate(context_data["search_results"][:5], 1):
                    timestamp_str = result.timestamp.strftime("%Y-%m-%d %H:%M") if isinstance(result.timestamp, datetime) else str(result.timestamp)
                    
                    # Extract app and window info from metadata
                    app_name = result.metadata.get("app_name", "Unknown App")
                    window_title = result.metadata.get("window_title", "")
                    
                    context_parts.append(f"\n{i}. **{app_name}** - {timestamp_str}")
                    if window_title:
                        context_parts.append(f"   Window: {window_title}")
                    
                    # Add content preview
                    content_preview = result.content[:300]
                    if len(result.content) > 300:
                        content_preview += "..."
                    context_parts.append(f"   Content: {content_preview}")
                    context_parts.append(f"   Relevance: {result.similarity_score:.2f}")
            
            # Add recent activity
            if context_data["recent_activity"]:
                context_parts.append("\n## Recent Activity:")
                for i, activity in enumerate(context_data["recent_activity"][:3], 1):
                    timestamp_str = activity.get("timestamp", "Unknown time")
                    app_name = activity.get("app_name", "Unknown")
                    window_title = activity.get("window_title", "")
                    description = activity.get("description", "")
                    
                    context_parts.append(f"\n{i}. **{app_name}** - {timestamp_str}")
                    if window_title:
                        context_parts.append(f"   Window: {window_title}")
                    if description:
                        context_parts.append(f"   Activity: {description}")
                    
                    # Add text preview if available
                    text_preview = activity.get("text_preview", "")
                    if text_preview:
                        context_parts.append(f"   Text: {text_preview}")
            
            # Add activity insights
            insights = context_data.get("insights", {})
            if insights.get("activity_summary"):
                context_parts.append(f"\n## Activity Summary:")
                context_parts.append(insights["activity_summary"])
                
                if insights.get("most_active_apps"):
                    apps = list(insights["most_active_apps"].keys())[:3]
                    context_parts.append(f"Most active apps: {', '.join(apps)}")
            
            if not context_parts:
                context_parts.append("No relevant screen capture data found for this query.")
            
            return "\n".join(context_parts), context_data["search_results"]
            
        except Exception as e:
            self.logger.error(f"Failed to get context for query: {e}")
            return "Unable to retrieve screen capture context.", []
    
    async def _get_recent_activity(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent activity from metadata database."""
        try:
            # Get recent screenshots with analysis
            recent = self.metadata_db.get_recent_screenshots(limit)
            
            results = []
            for screenshot in recent:
                # Get content analysis for each screenshot
                analysis = self.metadata_db.get_content_analysis(screenshot["id"])
                
                if analysis:
                    result = {
                        "id": screenshot["id"],
                        "timestamp": screenshot["timestamp"],
                        "content_type": analysis.get("content_type", "unknown"),
                        "description": analysis.get("description", ""),
                        "tags": analysis.get("tags", []),
                        "app_name": analysis.get("app_name", "Unknown"),
                        "window_title": analysis.get("window_title", "")
                    }
                    
                    # Add OCR text if available
                    ocr_result = self.metadata_db.get_ocr_result(screenshot["id"])
                    if ocr_result and ocr_result.get("text"):
                        result["text_preview"] = ocr_result["text"][:200]
                    
                    results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to get recent activity: {e}")
            return []
    
    @log_performance
    async def process_query(self, query: str) -> QueryResult:
        """
        Process a natural language query from the user.
        
        Args:
            query: User's question or command.
            
        Returns:
            QueryResult: Processed response with sources and confidence.
        """
        self.logger.debug(f"Processing query: {query}")
        
        try:
            # Add user query to conversation history
            self.conversation_history.add_turn("user", query)
            
            # Get relevant context from screen captures
            context, search_results = await self._get_context_for_query(query)
            
            # Build messages for LLM
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            # Add conversation history
            conversation_context = self.conversation_history.get_context(max_tokens=1000)
            for turn in conversation_context[:-1]:  # Exclude the current query
                messages.append(turn)
            
            # Add current query with context
            user_message = f"""Screen capture context for your query:

{context}

User query: {query}

Please provide a helpful response based on the screen capture data provided."""
            
            messages.append({"role": "user", "content": user_message})
            
            # Call LLM for response
            response_text = await self._generate_llm_response(messages)
            
            # Add assistant response to conversation history
            self.conversation_history.add_turn("assistant", response_text)
            
            # Extract sources from search results
            sources = [result.content_id for result in search_results[:5]]
            
            return QueryResult(
                query=query,
                response=response_text,
                confidence=0.8 if search_results else 0.5,
                sources=sources,
                metadata={
                    "provider": self.preferred_provider,
                    "context_items": len(search_results),
                    "conversation_turns": len(self.conversation_history.turns)
                },
                context_used=search_results[:5]
            )
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            error_response = f"I encountered an error processing your query: {str(e)}"
            
            return QueryResult(
                query=query,
                response=error_response,
                confidence=0.0,
                sources=[],
                metadata={"error": str(e)}
            )
    
    async def _generate_llm_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using available LLM."""
        if not self.preferred_provider:
            return "No AI providers available. Please configure API keys for Gemini, Claude, or OpenAI."
        
        try:
            # Convert messages to prompt for providers that need it
            if self.preferred_provider == "gemini":
                # Gemini prefers a single prompt
                prompt = self._messages_to_prompt(messages)
                response = await self.cloud_api.apis["gemini"].analyze_text(
                    prompt,
                    analysis_type="general"
                )
            else:
                # Other providers can handle message format better
                prompt = self._messages_to_prompt(messages)
                response = await self.cloud_api.analyze_text(
                    prompt,
                    analysis_type="general",
                    preferred_provider=self.preferred_provider
                )
            
            if response:
                return response.content
            else:
                return "I couldn't generate a response. Please try again."
                
        except Exception as e:
            self.logger.error(f"LLM response generation failed: {e}")
            return f"Failed to generate response: {str(e)}"
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to a single prompt string."""
        prompt_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts)
    
    def get_suggestions(self, context: str = "") -> List[str]:
        """
        Get suggested queries based on context and recent activity.
        
        Args:
            context: Current context or previous queries.
            
        Returns:
            List[str]: Suggested queries for the user.
        """
        base_suggestions = [
            "What was I working on today?",
            "Show me any Python code from the last hour",
            "Find all terminal commands I ran yesterday",
            "What websites did I visit this morning?",
            "Show me any error messages from today",
            "Find emails I was reading earlier",
            "What documents did I open recently?",
            "Show me my coding activity from this week"
        ]
        
        # Add context-aware suggestions if conversation history exists
        if self.conversation_history.turns:
            last_query = None
            for turn in reversed(self.conversation_history.turns):
                if turn.role == "user":
                    last_query = turn.content
                    break
            
            if last_query:
                # Add follow-up suggestions
                if "code" in last_query.lower():
                    base_suggestions.insert(0, "Show me more code examples")
                    base_suggestions.insert(1, "Find related code files")
                elif "error" in last_query.lower():
                    base_suggestions.insert(0, "Find similar errors")
                    base_suggestions.insert(1, "Show me the stack trace")
                elif "email" in last_query.lower():
                    base_suggestions.insert(0, "Show more emails from today")
                    base_suggestions.insert(1, "Find emails from this sender")
        
        return base_suggestions[:8]
    
    def clear_conversation(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
        self.logger.info("Conversation history cleared")
    
    async def get_chat_status(self) -> Dict[str, Any]:
        """Get current chat system status."""
        return {
            "available": bool(self.preferred_provider),
            "provider": self.preferred_provider,
            "available_providers": self.available_providers,
            "conversation_length": len(self.conversation_history.turns),
            "memory_stats": self.memory_system.get_memory_statistics()
        }