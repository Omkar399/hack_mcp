"""
Memory System component for Eidolon AI Personal Assistant

Handles storage, indexing, and retrieval of captured content with semantic
search capabilities, natural language queries, and RAG-based responses.
"""

import asyncio
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
from pathlib import Path

from ..utils.logging import get_component_logger, log_performance, log_exceptions
from ..utils.config import get_config
from ..storage.vector_db import VectorDatabase
from ..storage.metadata_db import MetadataDatabase
from ..models.cloud_api import CloudAPIManager
from ..models.decision_engine import DecisionEngine, AnalysisRequest, RoutingDecision


class SearchResult:
    """Represents a search result from the memory system."""
    
    def __init__(
        self,
        content_id: str,
        content: str,
        similarity_score: float,
        timestamp: datetime,
        metadata: Optional[Dict[str, Any]] = None,
        source_type: str = "screenshot"
    ):
        self.content_id = content_id
        self.content = content
        self.similarity_score = similarity_score
        self.timestamp = timestamp
        self.metadata = metadata or {}
        self.source_type = source_type
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content_id": self.content_id,
            "content": self.content,
            "similarity_score": self.similarity_score,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            "metadata": self.metadata,
            "source_type": self.source_type
        }


class QueryIntent:
    """Represents the intent and parameters of a natural language query."""
    
    def __init__(
        self,
        original_query: str,
        intent_type: str,
        search_terms: List[str],
        filters: Dict[str, Any],
        time_range: Optional[Dict[str, datetime]] = None,
        confidence: float = 0.0
    ):
        self.original_query = original_query
        self.intent_type = intent_type  # "search", "summarize", "analyze", "compare", etc.
        self.search_terms = search_terms
        self.filters = filters
        self.time_range = time_range
        self.confidence = confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_query": self.original_query,
            "intent_type": self.intent_type,
            "search_terms": self.search_terms,
            "filters": self.filters,
            "time_range": {
                "start": self.time_range["start"].isoformat() if self.time_range and "start" in self.time_range else None,
                "end": self.time_range["end"].isoformat() if self.time_range and "end" in self.time_range else None
            } if self.time_range else None,
            "confidence": self.confidence
        }


class MemoryResponse:
    """Represents a response from the memory system."""
    
    def __init__(
        self,
        query: str,
        response: str,
        search_results: List[SearchResult],
        query_intent: QueryIntent,
        generated_by: str = "local",
        confidence: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.query = query
        self.response = response
        self.search_results = search_results
        self.query_intent = query_intent
        self.generated_by = generated_by
        self.confidence = confidence
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "response": self.response,
            "search_results": [result.to_dict() for result in self.search_results],
            "query_intent": self.query_intent.to_dict(),
            "generated_by": self.generated_by,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class MemorySystem:
    """
    Advanced memory system with semantic search, natural language queries,
    and RAG-based response generation.
    """
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_component_logger("memory")
        
        # Initialize storage systems
        self.vector_db = VectorDatabase()
        self.metadata_db = MetadataDatabase()
        
        # Initialize AI components
        self.cloud_api = CloudAPIManager()
        self.decision_engine = DecisionEngine()
        
        # Query processing patterns
        self.intent_patterns = self._load_intent_patterns()
        
        self.logger.info("Memory system initialized with semantic search and RAG capabilities")
    
    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for intent recognition."""
        return {
            "search": [
                r"find|search|look for|show me",
                r"when did|what was|where is",
                r"list|display|get"
            ],
            "summarize": [
                r"summarize|summary|overview",
                r"what happened|recap|brief",
                r"main points|key events"
            ],
            "analyze": [
                r"analyze|analysis|understand",
                r"what does this mean|explain",
                r"insights|patterns|trends"
            ],
            "compare": [
                r"compare|difference|similar",
                r"vs|versus|between",
                r"which is better|pros and cons"
            ],
            "timeline": [
                r"timeline|chronology|sequence",
                r"what happened first|order of events",
                r"before|after|during"
            ]
        }
    
    @log_performance
    def store_content(
        self,
        screenshot_id: str,
        content_analysis: Dict[str, Any],
        extracted_text: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store content in both vector and metadata databases.
        
        Args:
            screenshot_id: Unique identifier for the screenshot
            content_analysis: Content analysis results
            extracted_text: OCR extracted text
            metadata: Additional metadata
            
        Returns:
            bool: True if stored successfully
        """
        try:
            # Store in vector database for semantic search
            vector_success = self.vector_db.store_content(
                screenshot_id, content_analysis, extracted_text, metadata
            )
            
            # Also ensure it's stored in metadata database
            if extracted_text and vector_success:
                # Store OCR results
                ocr_result = {
                    "text": extracted_text,
                    "confidence": content_analysis.get("confidence", 0.0),
                    "method": "phase4_enhanced",
                    "word_count": len(extracted_text.split()) if extracted_text else 0
                }
                self.metadata_db.store_ocr_result(screenshot_id, ocr_result)
                
                # Store content analysis
                self.metadata_db.store_content_analysis(screenshot_id, content_analysis)
            
            return vector_success
            
        except Exception as e:
            self.logger.error(f"Failed to store content in memory system: {e}")
            return False
    
    @log_performance
    def parse_natural_language_query(self, query: str) -> QueryIntent:
        """
        Parse a natural language query to extract intent and parameters.
        
        Args:
            query: Natural language query string
            
        Returns:
            QueryIntent: Parsed query intent and parameters
        """
        import re
        from datetime import datetime, timedelta
        
        query_lower = query.lower()
        
        # Determine intent type
        intent_type = "search"  # Default
        intent_confidence = 0.5
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    intent_type = intent
                    intent_confidence = 0.8
                    break
            if intent_confidence > 0.5:
                break
        
        # Extract search terms (remove common words)
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "up", "about", "into", "through", "during",
            "before", "after", "above", "below", "between", "among", "under", "over",
            "is", "was", "are", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "may", "might",
            "can", "must", "shall", "what", "when", "where", "who", "why", "how",
            "find", "search", "look", "show", "me", "get", "list", "display"
        }
        
        words = re.findall(r'\b\w+\b', query_lower)
        search_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Extract filters
        filters = {}
        
        # Content type filters
        content_types = ["code", "email", "document", "browser", "terminal", "error", "chat"]
        for content_type in content_types:
            if content_type in query_lower:
                filters["content_type"] = content_type
                break
        
        # Time-based filters
        time_range = None
        
        # Today, yesterday, this week, etc.
        if "today" in query_lower:
            today = datetime.now()
            time_range = {
                "start": today.replace(hour=0, minute=0, second=0, microsecond=0),
                "end": today
            }
        elif "yesterday" in query_lower:
            yesterday = datetime.now() - timedelta(days=1)
            time_range = {
                "start": yesterday.replace(hour=0, minute=0, second=0, microsecond=0),
                "end": yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
            }
        elif "this week" in query_lower or "past week" in query_lower:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            time_range = {"start": start_date, "end": end_date}
        elif "this month" in query_lower or "past month" in query_lower:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            time_range = {"start": start_date, "end": end_date}
        
        # Extract specific dates (basic patterns)
        date_match = re.search(r'(\d{1,2})/(\d{1,2})/(\d{2,4})', query)
        if date_match:
            try:
                month, day, year = date_match.groups()
                year = int(year)
                if year < 100:
                    year += 2000
                target_date = datetime(year, int(month), int(day))
                time_range = {
                    "start": target_date.replace(hour=0, minute=0, second=0, microsecond=0),
                    "end": target_date.replace(hour=23, minute=59, second=59, microsecond=999999)
                }
            except ValueError:
                pass  # Invalid date format
        
        return QueryIntent(
            original_query=query,
            intent_type=intent_type,
            search_terms=search_terms,
            filters=filters,
            time_range=time_range,
            confidence=intent_confidence
        )
    
    @log_performance
    async def semantic_search(
        self,
        query: str,
        n_results: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.6
    ) -> List[SearchResult]:
        """
        Perform semantic search across stored content.
        
        Args:
            query: Search query
            n_results: Maximum number of results
            filters: Optional filters to apply
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Use vector database for semantic search
            search_kwargs = {
                "n_results": n_results,
                "include_metadata": True
            }
            
            if filters:
                search_kwargs.update(filters)
            
            vector_results = self.vector_db.semantic_search(query, **search_kwargs)
            
            # Convert to SearchResult objects
            search_results = []
            for result in vector_results:
                if result["similarity"] >= similarity_threshold:
                    metadata = result.get("metadata", {})
                    
                    # Parse timestamp
                    timestamp = metadata.get("timestamp")
                    if isinstance(timestamp, str):
                        try:
                            timestamp = datetime.fromisoformat(timestamp)
                        except ValueError:
                            timestamp = datetime.now()
                    elif not isinstance(timestamp, datetime):
                        timestamp = datetime.now()
                    
                    search_results.append(SearchResult(
                        content_id=metadata.get("screenshot_id", result["id"]),
                        content=result["document"],
                        similarity_score=result["similarity"],
                        timestamp=timestamp,
                        metadata=metadata,
                        source_type="screenshot"
                    ))
            
            self.logger.debug(f"Semantic search for '{query}' returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return []
    
    @log_performance
    async def hybrid_search(
        self,
        query: str,
        n_results: int = 10,
        semantic_weight: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining semantic and keyword matching.
        
        Args:
            query: Search query
            n_results: Number of results
            semantic_weight: Weight for semantic vs keyword matching
            filters: Optional filters
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Use vector database hybrid search
            vector_results = self.vector_db.hybrid_search(
                query,
                n_results=n_results,
                semantic_weight=semantic_weight,
                keyword_weight=1.0 - semantic_weight,
                **(filters or {})
            )
            
            # Convert to SearchResult objects
            search_results = []
            for result in vector_results:
                metadata = result.get("metadata", {})
                
                # Parse timestamp
                timestamp = metadata.get("timestamp")
                if isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp)
                    except ValueError:
                        timestamp = datetime.now()
                elif not isinstance(timestamp, datetime):
                    timestamp = datetime.now()
                
                search_results.append(SearchResult(
                    content_id=metadata.get("screenshot_id", result["id"]),
                    content=result["document"],
                    similarity_score=result.get("combined_score", result["similarity"]),
                    timestamp=timestamp,
                    metadata=metadata,
                    source_type="screenshot"
                ))
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Hybrid search failed: {e}")
            return []
    
    @log_performance
    async def generate_rag_response(
        self,
        query: str,
        search_results: List[SearchResult],
        max_context_length: int = 2000
    ) -> Optional[str]:
        """
        Generate a RAG (Retrieval-Augmented Generation) response using search results.
        
        Args:
            query: Original user query
            search_results: Retrieved relevant content
            max_context_length: Maximum length of context to include
            
        Returns:
            Generated response string or None if generation fails
        """
        if not search_results:
            return "I couldn't find any relevant information for your query."
        
        try:
            # Prepare context from search results
            context_parts = []
            total_length = 0
            
            for result in search_results[:5]:  # Use top 5 results
                content = result.content
                if total_length + len(content) > max_context_length:
                    # Truncate to fit within limit
                    remaining_space = max_context_length - total_length
                    if remaining_space > 100:  # Only add if there's meaningful space
                        content = content[:remaining_space - 3] + "..."
                        context_parts.append(content)
                    break
                
                context_parts.append(content)
                total_length += len(content)
            
            context = "\n\n".join(context_parts)
            
            # Create RAG prompt
            rag_prompt = f"""Based on the following information from the user's activity history, please provide a helpful and accurate response to their question.

Context:
{context}

User Question: {query}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to fully answer the question, please say so and provide what information is available."""
            
            # Determine which AI service to use
            request = AnalysisRequest(
                content_type="text",
                text_content=rag_prompt,
                metadata={"query_type": "rag", "context_length": len(context)},
                user_preferences={"importance": 0.7}  # RAG queries are generally important
            )
            
            available_providers = self.cloud_api.get_available_providers()
            decision = self.decision_engine.make_routing_decision(request, available_providers)
            
            if decision.use_cloud and decision.provider:
                # Use cloud AI for response generation
                cloud_response = await self.cloud_api.analyze_text(
                    rag_prompt,
                    analysis_type="general",
                    preferred_provider=decision.provider
                )
                
                if cloud_response:
                    self.decision_engine.record_analysis_result(
                        decision,
                        actual_cost=sum(cloud_response.usage.values()) * 0.001,  # Rough estimate
                        success=True
                    )
                    return cloud_response.content
            
            # Fallback to local/basic response
            self.decision_engine.record_analysis_result(decision, success=True)
            return self._generate_basic_rag_response(query, search_results)
            
        except Exception as e:
            self.logger.error(f"RAG response generation failed: {e}")
            return f"I found relevant information but encountered an error generating the response: {str(e)}"
    
    def _generate_basic_rag_response(self, query: str, search_results: List[SearchResult]) -> str:
        """Generate a basic RAG response without cloud AI."""
        if not search_results:
            return "I couldn't find any relevant information for your query."
        
        # Simple response generation based on search results
        response_parts = [
            f"Based on your activity history, I found {len(search_results)} relevant items:"
        ]
        
        for i, result in enumerate(search_results[:3], 1):
            timestamp_str = result.timestamp.strftime("%Y-%m-%d %H:%M") if isinstance(result.timestamp, datetime) else str(result.timestamp)
            content_preview = result.content[:150]
            if len(result.content) > 150:
                content_preview += "..."
            
            response_parts.append(
                f"{i}. {timestamp_str}: {content_preview} (similarity: {result.similarity_score:.2f})"
            )
        
        if len(search_results) > 3:
            response_parts.append(f"... and {len(search_results) - 3} more results")
        
        return "\n\n".join(response_parts)
    
    @log_performance
    async def process_natural_language_query(self, query: str) -> MemoryResponse:
        """
        Process a natural language query end-to-end.
        
        Args:
            query: Natural language query string
            
        Returns:
            MemoryResponse with search results and generated response
        """
        try:
            # Parse the query to understand intent
            query_intent = self.parse_natural_language_query(query)
            
            # Perform search based on intent
            if query_intent.intent_type in ["search", "find"]:
                search_results = await self.semantic_search(
                    " ".join(query_intent.search_terms),
                    n_results=10,
                    filters=query_intent.filters
                )
            elif query_intent.intent_type in ["summarize", "analyze", "compare"]:
                # For complex queries, use hybrid search
                search_results = await self.hybrid_search(
                    " ".join(query_intent.search_terms),
                    n_results=15,
                    filters=query_intent.filters
                )
            else:
                # Default to semantic search
                search_results = await self.semantic_search(
                    query,
                    n_results=10,
                    filters=query_intent.filters
                )
            
            # Generate response based on intent
            if query_intent.intent_type in ["summarize", "analyze", "compare"] and search_results:
                # Use RAG for complex queries
                response = await self.generate_rag_response(query, search_results)
                generated_by = "rag"
                confidence = 0.8
            else:
                # Simple response for basic searches
                response = self._generate_basic_rag_response(query, search_results)
                generated_by = "local"
                confidence = 0.6
            
            return MemoryResponse(
                query=query,
                response=response or "I couldn't generate a response for your query.",
                search_results=search_results,
                query_intent=query_intent,
                generated_by=generated_by,
                confidence=confidence,
                metadata={
                    "search_count": len(search_results),
                    "avg_similarity": sum(r.similarity_score for r in search_results) / len(search_results) if search_results else 0.0
                }
            )
            
        except Exception as e:
            self.logger.error(f"Natural language query processing failed: {e}")
            return MemoryResponse(
                query=query,
                response=f"Sorry, I encountered an error processing your query: {str(e)}",
                search_results=[],
                query_intent=QueryIntent(query, "error", [], {}, confidence=0.0),
                generated_by="error",
                confidence=0.0
            )
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics."""
        try:
            # Vector database stats
            vector_stats = self.vector_db.get_statistics()
            
            # Metadata database stats
            metadata_stats = self.metadata_db.get_statistics()
            
            # Decision engine stats
            decision_stats = self.decision_engine.get_decision_stats()
            
            # Cloud API usage
            cloud_usage = self.cloud_api.get_usage_stats()
            
            return {
                "vector_database": vector_stats,
                "metadata_database": metadata_stats,
                "decision_engine": decision_stats,
                "cloud_usage": cloud_usage,
                "total_searchable_content": vector_stats.get("total_documents", 0),
                "capabilities": {
                    "semantic_search": True,
                    "natural_language_queries": True,
                    "rag_responses": len(self.cloud_api.get_available_providers()) > 0,
                    "hybrid_search": True,
                    "intent_recognition": True
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get memory statistics: {e}")
            return {"error": str(e)}
    
    async def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up old data from all storage systems."""
        try:
            # Cleanup vector database
            vector_removed = self.vector_db.cleanup_old_entries(days_to_keep)
            
            # Cleanup metadata database
            metadata_removed = self.metadata_db.cleanup_old_screenshots(days_to_keep)
            
            return {
                "vector_entries_removed": vector_removed,
                "metadata_entries_removed": metadata_removed,
                "total_removed": vector_removed + metadata_removed
            }
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            return {"error": str(e)}
    
    async def get_conversation_context(
        self,
        query: str,
        time_window_hours: int = 24,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Get enriched context for conversational queries.
        
        This method is optimized for chat interactions, providing
        both relevant search results and recent activity context.
        
        Args:
            query: User's query
            time_window_hours: Hours of recent activity to include
            max_results: Maximum number of results
            
        Returns:
            Dictionary with search results, recent activity, and insights
        """
        try:
            # Parse query intent
            query_intent = self.parse_natural_language_query(query)
            
            # Get semantic search results
            search_results = await self.semantic_search(
                query,
                n_results=max_results,
                filters=query_intent.filters
            )
            
            # Get recent activity within time window
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=time_window_hours)
            
            recent_activity = []
            recent_screenshots = self.metadata_db.get_screenshots_by_time_range(
                start_time, end_time, limit=20
            )
            
            for screenshot in recent_screenshots:
                # Get analysis and OCR for each
                analysis = self.metadata_db.get_content_analysis(screenshot["id"])
                ocr_result = self.metadata_db.get_ocr_result(screenshot["id"])
                
                if analysis or ocr_result:
                    activity = {
                        "timestamp": screenshot["timestamp"],
                        "content_type": analysis.get("content_type", "unknown") if analysis else "unknown",
                        "app_name": analysis.get("app_name", "Unknown") if analysis else "Unknown",
                        "window_title": analysis.get("window_title", "") if analysis else "",
                        "description": analysis.get("description", "") if analysis else "",
                        "text_preview": ocr_result.get("text", "")[:200] if ocr_result else ""
                    }
                    recent_activity.append(activity)
            
            # Generate activity insights
            insights = self._generate_activity_insights(recent_activity, search_results)
            
            return {
                "search_results": search_results,
                "recent_activity": recent_activity[:10],  # Limit to 10 most recent
                "query_intent": query_intent.to_dict(),
                "insights": insights,
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get conversation context: {e}")
            return {
                "search_results": [],
                "recent_activity": [],
                "query_intent": {},
                "insights": {},
                "error": str(e)
            }
    
    def _generate_activity_insights(
        self,
        recent_activity: List[Dict[str, Any]],
        search_results: List[SearchResult]
    ) -> Dict[str, Any]:
        """Generate insights from activity data."""
        insights = {
            "most_active_apps": {},
            "content_types": {},
            "activity_summary": "",
            "search_relevance": 0.0
        }
        
        try:
            # Count app usage
            app_counts = {}
            content_type_counts = {}
            
            for activity in recent_activity:
                app = activity.get("app_name", "Unknown")
                app_counts[app] = app_counts.get(app, 0) + 1
                
                content_type = activity.get("content_type", "unknown")
                content_type_counts[content_type] = content_type_counts.get(content_type, 0) + 1
            
            # Sort by frequency
            insights["most_active_apps"] = dict(sorted(
                app_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5])
            
            insights["content_types"] = dict(sorted(
                content_type_counts.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            
            # Calculate search relevance
            if search_results:
                avg_similarity = sum(r.similarity_score for r in search_results) / len(search_results)
                insights["search_relevance"] = round(avg_similarity, 2)
            
            # Generate summary
            if recent_activity:
                total_items = len(recent_activity)
                top_app = max(app_counts.items(), key=lambda x: x[1])[0] if app_counts else "Unknown"
                insights["activity_summary"] = f"Found {total_items} activities. Most used: {top_app}"
            else:
                insights["activity_summary"] = "No recent activity found"
            
        except Exception as e:
            self.logger.warning(f"Failed to generate insights: {e}")
        
        return insights