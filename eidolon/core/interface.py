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
        
        # Context tracking for follow-up conversations
        self.last_query_context = None
        self.conversation_topics = []  # Track conversation topics
        
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
    
    async def _get_enhanced_context_for_query(self, query: str, memory_response) -> Tuple[str, List[SearchResult]]:
        """Get enhanced context including both screen captures and activity timeline."""
        try:
            # Start with the memory response results
            search_results = memory_response.search_results
            
            # Build comprehensive context string
            context_parts = []
            
            # Add query intent information
            if memory_response.query_intent:
                intent_info = []
                if memory_response.query_intent.intent_type != 'search':
                    intent_info.append(f"Query type: {memory_response.query_intent.intent_type}")
                
                if memory_response.query_intent.filters:
                    filters_str = ", ".join([f"{k}: {v}" for k, v in memory_response.query_intent.filters.items()])
                    intent_info.append(f"Filters: {filters_str}")
                
                if memory_response.query_intent.time_range:
                    tr = memory_response.query_intent.time_range
                    start_str = tr['start'].strftime("%Y-%m-%d %H:%M") if hasattr(tr['start'], 'strftime') else str(tr['start'])
                    end_str = tr['end'].strftime("%Y-%m-%d %H:%M") if hasattr(tr['end'], 'strftime') else str(tr['end'])
                    intent_info.append(f"Time range: {start_str} to {end_str}")
                
                if intent_info:
                    context_parts.append(f"## Query Analysis:\n{'; '.join(intent_info)}")
            
            # Separate activity timeline results from regular search results
            activity_results = [r for r in search_results if r.source_type == "activity_timeline"]
            screen_results = [r for r in search_results if r.source_type != "activity_timeline"]
            
            # Add activity timeline context
            if activity_results:
                context_parts.append("## Recent Activity Timeline:")
                for i, result in enumerate(activity_results[:5], 1):
                    timestamp_str = result.timestamp.strftime("%Y-%m-%d %H:%M") if hasattr(result.timestamp, 'strftime') else str(result.timestamp)
                    
                    platform = result.metadata.get('platform', 'Unknown')
                    activity_type = result.metadata.get('activity_type', 'activity')
                    title = result.metadata.get('title', '')
                    
                    context_parts.append(f"\n{i}. **{platform}** {activity_type} - {timestamp_str}")
                    if title:
                        context_parts.append(f"   Title: {title}")
                    
                    # Add relevant metadata
                    url = result.metadata.get('url', '')
                    domain = result.metadata.get('domain', '')
                    if url:
                        context_parts.append(f"   URL: {url}")
                    elif domain:
                        context_parts.append(f"   Domain: {domain}")
                    
                    # Add content preview
                    content_preview = result.content[:200]
                    if len(result.content) > 200:
                        content_preview += "..."
                    context_parts.append(f"   Details: {content_preview}")
                    context_parts.append(f"   Relevance: {result.similarity_score:.2f}")
            
            # Add regular screen capture context
            if screen_results:
                context_parts.append("## Screen Capture Context:")
                for i, result in enumerate(screen_results[:5], 1):
                    timestamp_str = result.timestamp.strftime("%Y-%m-%d %H:%M") if hasattr(result.timestamp, 'strftime') else str(result.timestamp)
                    
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
            
            # Add activity insights from memory response metadata
            if memory_response.metadata:
                if memory_response.metadata.get('search_count', 0) > 0:
                    context_parts.append(f"\n## Search Summary:")
                    context_parts.append(f"Found {memory_response.metadata['search_count']} relevant items")
                    if memory_response.metadata.get('avg_similarity'):
                        context_parts.append(f"Average relevance: {memory_response.metadata['avg_similarity']:.2f}")
            
            if not context_parts:
                context_parts.append("No relevant context found for this query.")
            
            return "\n".join(context_parts), search_results
            
        except Exception as e:
            self.logger.error(f"Failed to get enhanced context for query: {e}")
            # Fallback to original method
            return await self._get_context_for_query(query)
    
    def _update_conversation_context(self, query: str, memory_response, search_results: List):
        """Update conversation context for better follow-up handling."""
        try:
            from datetime import datetime
            
            # Store context from this query
            self.last_query_context = {
                'query': query,
                'timestamp': datetime.now(),
                'query_intent': memory_response.query_intent.to_dict() if memory_response.query_intent else {},
                'search_results': search_results[:3],  # Keep top 3 for context
                'generated_by': memory_response.generated_by,
                'platforms_mentioned': [],
                'domains_mentioned': [],
                'time_range': None
            }
            
            # Extract conversation topics and context
            query_lower = query.lower()
            
            # Track platforms mentioned
            platforms = ['youtube', 'netflix', 'twitter', 'facebook', 'instagram', 'browser']
            for platform in platforms:
                if platform in query_lower:
                    self.last_query_context['platforms_mentioned'].append(platform)
            
            # Track domains from results
            for result in search_results[:5]:
                if hasattr(result, 'metadata') and result.metadata:
                    domain = result.metadata.get('domain', '')
                    platform = result.metadata.get('platform', '')
                    if domain and domain not in self.last_query_context['domains_mentioned']:
                        self.last_query_context['domains_mentioned'].append(domain)
                    if platform and platform not in self.last_query_context['platforms_mentioned']:
                        self.last_query_context['platforms_mentioned'].append(platform)
            
            # Track time range if specified
            if memory_response.query_intent and memory_response.query_intent.time_range:
                self.last_query_context['time_range'] = memory_response.query_intent.time_range
            
            # Update conversation topics (keep last 5)
            topic = self._extract_conversation_topic(query, memory_response)
            if topic:
                self.conversation_topics.append({
                    'topic': topic,
                    'timestamp': datetime.now(),
                    'query': query
                })
                # Keep only last 5 topics
                self.conversation_topics = self.conversation_topics[-5:]
                
        except Exception as e:
            self.logger.warning(f"Failed to update conversation context: {e}")
    
    def _extract_conversation_topic(self, query: str, memory_response) -> Optional[str]:
        """Extract the main topic from a query for conversation tracking."""
        query_lower = query.lower()
        
        # Domain-specific topics
        if any(term in query_lower for term in ['youtube', 'video']):
            return 'youtube_videos'
        elif any(term in query_lower for term in ['netflix', 'movie', 'show']):
            return 'netflix_content'
        elif any(term in query_lower for term in ['website', 'visited', 'browsed']):
            return 'web_browsing'
        elif any(term in query_lower for term in ['code', 'programming', 'python', 'javascript']):
            return 'coding_activity'
        elif any(term in query_lower for term in ['error', 'bug', 'debug']):
            return 'error_debugging'
        elif any(term in query_lower for term in ['email', 'message']):
            return 'communication'
        elif any(term in query_lower for term in ['document', 'file', 'pdf']):
            return 'document_work'
        elif any(term in query_lower for term in ['terminal', 'command', 'shell']):
            return 'terminal_work'
        
        return None
    
    def _get_context_aware_suggestions(self) -> List[str]:
        """Get suggestions based on conversation context and recent topics."""
        suggestions = []
        
        if not self.last_query_context:
            return self.get_suggestions()
        
        # Get context from last query
        last_platforms = self.last_query_context.get('platforms_mentioned', [])
        last_domains = self.last_query_context.get('domains_mentioned', [])
        last_time_range = self.last_query_context.get('time_range')
        
        # Platform-specific follow-ups
        if 'youtube' in last_platforms:
            suggestions.extend([
                "Show me more YouTube videos from the same channel",
                "What YouTube videos did I watch before that one?",
                "Find similar YouTube content"
            ])
        
        if 'netflix' in last_platforms:
            suggestions.extend([
                "What other episodes of this show have I watched?",
                "Show me other Netflix shows I've been watching",
                "Find similar Netflix content"
            ])
        
        if 'browser' in last_platforms or last_domains:
            suggestions.extend([
                "Show me other websites from the same domain",
                "What else was I browsing around that time?",
                "Find related web activity"
            ])
        
        # Time-based follow-ups
        if last_time_range:
            suggestions.extend([
                "What else did I do during that time period?",
                "Show me activity from the same day",
                "Expand the time range to see more"
            ])
        
        # Topic continuation suggestions
        recent_topics = [topic['topic'] for topic in self.conversation_topics[-3:]]
        
        if 'youtube_videos' in recent_topics:
            suggestions.append("Compare my YouTube watching habits over time")
        
        if 'coding_activity' in recent_topics:
            suggestions.append("Show me my overall programming productivity")
        
        if 'web_browsing' in recent_topics:
            suggestions.append("What are my most visited websites?")
        
        # Remove duplicates and limit
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in seen:
                seen.add(suggestion)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions[:8]
    
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
            
            # Use enhanced memory system for comprehensive query processing
            memory_response = await self.memory_system.process_natural_language_query(query)
            
            # Check if we got a specialized response
            if memory_response.generated_by == "specialized":
                # Use the specialized response directly
                response_text = memory_response.response
                confidence = memory_response.confidence
                sources = [result.content_id for result in memory_response.search_results[:5]]
                
                # Add assistant response to conversation history
                self.conversation_history.add_turn("assistant", response_text)
                
                # Update conversation context for follow-ups
                self._update_conversation_context(query, memory_response, memory_response.search_results)
                
                return QueryResult(
                    query=query,
                    response=response_text,
                    confidence=confidence,
                    sources=sources,
                    metadata={
                        "provider": "enhanced_memory",
                        "generated_by": memory_response.generated_by,
                        "context_items": len(memory_response.search_results),
                        "conversation_turns": len(self.conversation_history.turns),
                        "query_intent": memory_response.query_intent.to_dict(),
                        "specialized_handler": True
                    },
                    context_used=memory_response.search_results[:5]
                )
            
            # For non-specialized queries, fall back to LLM with enhanced context
            context, search_results = await self._get_enhanced_context_for_query(query, memory_response)
            
            # Build messages for LLM
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            # Add conversation history
            conversation_context = self.conversation_history.get_context(max_tokens=1000)
            for turn in conversation_context[:-1]:  # Exclude the current query
                messages.append(turn)
            
            # Add current query with enhanced context and conversation awareness
            contextual_query = self._enhance_query_with_conversation_context(query, conversation_context)
            
            user_message = f"""Enhanced screen capture and activity context for your query:

{context}

User query: {query}
Contextual interpretation: {contextual_query}

Please provide a helpful response based on:
1. The enhanced screen capture and activity context provided above
2. Our conversation history for follow-up questions 
3. Be specific with timestamps, details, and actionable insights
4. If this is a follow-up question (like "at what time?" or "what was useful about it?"), reference the previous topic

The context includes comprehensive screenshot data, OCR text, and metadata for accurate responses."""
            
            messages.append({"role": "user", "content": user_message})
            
            # Call LLM for response
            response_text = await self._generate_llm_response(messages)
            
            # Add assistant response to conversation history
            self.conversation_history.add_turn("assistant", response_text)
            
            # Update conversation context for follow-ups
            self._update_conversation_context(query, memory_response, search_results)
            
            # Extract sources from search results
            sources = [result.content_id for result in search_results[:5]]
            
            return QueryResult(
                query=query,
                response=response_text,
                confidence=0.8 if search_results else 0.5,
                sources=sources,
                metadata={
                    "provider": self.preferred_provider,
                    "generated_by": memory_response.generated_by,
                    "context_items": len(search_results),
                    "conversation_turns": len(self.conversation_history.turns),
                    "query_intent": memory_response.query_intent.to_dict(),
                    "enhanced_context": True
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
        # Try context-aware suggestions first if we have conversation context
        if self.last_query_context:
            context_suggestions = self._get_context_aware_suggestions()
            if context_suggestions:
                return context_suggestions
        
        # Enhanced suggestions that showcase the new temporal and domain-specific capabilities
        base_suggestions = [
            "What is the last YouTube video I watched?",
            "Show me Netflix content I watched this week",
            "What websites did I visit today?",
            "What was I working on yesterday?",
            "Find the last document I opened",
            "Show me recent social media activity",
            "What did I do this morning?",
            "Find Python code from the past 2 hours",
            "Show me all YouTube videos from last weekend",
            "What Netflix show was I watching recently?",
            "List websites I visited earlier today",
            "Find terminal commands from yesterday"
        ]
        
        # Add context-aware suggestions if conversation history exists
        if self.conversation_history.turns:
            last_query = None
            for turn in reversed(self.conversation_history.turns):
                if turn.role == "user":
                    last_query = turn.content
                    break
            
            if last_query:
                query_lower = last_query.lower()
                
                # Domain-specific follow-ups
                if any(term in query_lower for term in ["youtube", "video"]):
                    base_suggestions.insert(0, "Show me more YouTube videos I watched")
                    base_suggestions.insert(1, "What YouTube channel do I watch most?")
                elif any(term in query_lower for term in ["netflix", "movie", "show"]):
                    base_suggestions.insert(0, "What other Netflix shows have I watched?")
                    base_suggestions.insert(1, "Show me Netflix activity from this month")
                elif any(term in query_lower for term in ["website", "visited", "browsed"]):
                    base_suggestions.insert(0, "Show me more websites I visited today")
                    base_suggestions.insert(1, "What domains do I visit most often?")
                elif "code" in query_lower:
                    base_suggestions.insert(0, "Show me more code from the same project")
                    base_suggestions.insert(1, "Find related programming activity")
                elif "error" in query_lower:
                    base_suggestions.insert(0, "Find similar errors from this week")
                    base_suggestions.insert(1, "Show me debugging activity")
                
                # Temporal follow-ups
                if any(term in query_lower for term in ["today", "yesterday", "recent"]):
                    base_suggestions.insert(0, "What did I do earlier this week?")
                    base_suggestions.insert(1, "Show me activity from last month")
        
        return base_suggestions[:10]
    
    def clear_conversation(self) -> None:
        """Clear conversation history and context."""
        self.conversation_history.clear()
        self.last_query_context = None
        self.conversation_topics = []
        self.logger.info("Conversation history and context cleared")
    
    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search through captured content using the memory system."""
        try:
            # Use the memory system for semantic search
            search_results = await self.memory_system.semantic_search(
                query=query,
                n_results=limit,
                similarity_threshold=0.5
            )
            
            # Convert SearchResult objects to dictionaries for CLI compatibility
            results = []
            for result in search_results:
                results.append({
                    "title": f"Screenshot {result.content_id[:8]}...",
                    "content": result.content[:200] + "..." if len(result.content) > 200 else result.content,
                    "timestamp": result.timestamp.strftime("%Y-%m-%d %H:%M:%S") if isinstance(result.timestamp, datetime) else str(result.timestamp),
                    "similarity_score": result.similarity_score,
                    "source_type": result.source_type
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    async def chat(self, message: str) -> str:
        """Process a chat message and return response with enhanced capabilities."""
        try:
            # Process the query with enhanced system
            result = await self.process_query(message)
            
            # Return the response (conversation history is handled in process_query)
            return result.response
            
        except Exception as e:
            self.logger.error(f"Chat failed: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def get_status(self) -> Dict[str, Any]:
        """Get interface status for CLI compatibility."""
        return {
            "available": bool(self.preferred_provider),
            "provider": self.preferred_provider,
            "conversation_turns": len(self.conversation_history.turns)
        }
    
    async def get_chat_status(self) -> Dict[str, Any]:
        """Get current chat system status."""
        return {
            "available": bool(self.preferred_provider),
            "provider": self.preferred_provider,
            "available_providers": self.available_providers,
            "conversation_length": len(self.conversation_history.turns),
            "memory_stats": self.memory_system.get_memory_statistics()
        }
    
    def _enhance_query_with_conversation_context(self, query: str, conversation_context: List[Dict[str, str]]) -> str:
        """
        Enhance a query with conversation context for follow-up questions.
        
        Args:
            query: Current user query
            conversation_context: Previous conversation turns
            
        Returns:
            Enhanced query with contextual understanding
        """
        query_lower = query.lower().strip()
        
        # Check if this is a follow-up question
        follow_up_patterns = [
            "at what time", "what time", "when exactly", "what was the time",
            "what was useful", "what was the usefulness", "how useful", "was it useful",
            "what did i learn", "what was helpful", "what was good about",
            "tell me more", "more details", "explain more", "elaborate",
            "before that", "after that", "what happened next", "what came before",
            "how long", "duration", "length of time",
            "where was", "which website", "what site", "what page",
            "who was", "what channel", "which user", "what author"
        ]
        
        is_follow_up = any(pattern in query_lower for pattern in follow_up_patterns)
        
        if not is_follow_up or not conversation_context:
            return query  # Not a follow-up, return original query
        
        # Extract the last assistant response for context
        last_response = None
        for turn in reversed(conversation_context):
            if turn.get("role") == "assistant":
                last_response = turn.get("content", "")
                break
        
        if not last_response:
            return query
        
        # Build enhanced contextual query
        enhanced_parts = [f"Follow-up question about: {last_response[:200]}..."]
        
        # Add specific context based on question type
        if any(time_q in query_lower for time_q in ["time", "when"]):
            enhanced_parts.append("User wants specific timestamp information about the previously mentioned item.")
            
        elif any(useful_q in query_lower for useful_q in ["useful", "helpful", "good", "learn"]):
            enhanced_parts.append("User wants analysis of value/usefulness of the previously mentioned content.")
            
        elif any(detail_q in query_lower for detail_q in ["more", "details", "elaborate"]):
            enhanced_parts.append("User wants additional details about the previously mentioned topic.")
            
        elif any(nav_q in query_lower for nav_q in ["before", "after", "next"]):
            enhanced_parts.append("User wants temporal navigation relative to the previously mentioned activity.")
        
        enhanced_parts.append(f"Current query: {query}")
        
        return " | ".join(enhanced_parts)