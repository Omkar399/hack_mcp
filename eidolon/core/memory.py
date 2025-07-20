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
from ..integrations.fastmcp import get_fastmcp


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
        self.fastmcp = None  # Lazy load FastMCP
        
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
            ],
            "terminal": [
                r"terminal|command|cmd|shell",
                r"what.*command|last.*command",
                r"executed|ran.*command"
            ],
            "development": [
                r"implement|coded|built|developed",
                r"feature|function|class|method",
                r"programming|coding|development"
            ],
            "error_analysis": [
                r"wrong|error|failed|bug",
                r"what.*went.*wrong|what.*failed",
                r"debug|debugging|fix"
            ],
            "git": [
                r"git|clone|commit|push|pull",
                r"repository|repo|branch|merge",
                r"version.*control"
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
        
        # Enhanced temporal parsing with domain context
        enhanced_time_info = self._parse_enhanced_temporal_expressions(query, time_range)
        if enhanced_time_info:
            time_range = enhanced_time_info.get('time_range', time_range)
            if enhanced_time_info.get('platform_filter'):
                filters['platform'] = enhanced_time_info['platform_filter']
            if enhanced_time_info.get('domain_filter'):
                filters['domain'] = enhanced_time_info['domain_filter']
            if enhanced_time_info.get('activity_type'):
                filters['activity_type'] = enhanced_time_info['activity_type']
        
        return QueryIntent(
            original_query=query,
            intent_type=intent_type,
            search_terms=search_terms,
            filters=filters,
            time_range=time_range,
            confidence=intent_confidence
        )
    
    def _parse_enhanced_temporal_expressions(self, query: str, existing_time_range: Optional[Dict[str, datetime]]) -> Optional[Dict[str, Any]]:
        """
        Enhanced temporal expression parser for complex time queries.
        
        Args:
            query: Natural language query
            existing_time_range: Already parsed time range
            
        Returns:
            Dictionary with enhanced temporal information
        """
        import re
        from datetime import datetime, timedelta
        import calendar
        
        query_lower = query.lower()
        result = {}
        
        # Platform/domain detection patterns
        platform_patterns = {
            'youtube': ['youtube', 'video', 'watched', 'channel'],
            'netflix': ['netflix', 'movie', 'show', 'episode', 'series'],
            'browser': ['website', 'browsed', 'visited', 'webpage', 'url'],
            'social_media': ['twitter', 'facebook', 'instagram', 'social'],
            'terminal': ['terminal', 'command', 'cmd', 'shell', 'bash', 'zsh'],
            'ide': ['vscode', 'code editor', 'ide', 'vim', 'sublime', 'pycharm'],
            'development': ['coding', 'programming', 'development', 'implementing']
        }
        
        # Detect platform from query
        for platform, keywords in platform_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                result['platform_filter'] = platform
                break
        
        # Activity type detection
        activity_patterns = {
            'video_view': ['watched', 'viewing', 'video', 'movie', 'show'],
            'webpage_visit': ['visited', 'browsed', 'website', 'page'],
            'document_open': ['opened', 'document', 'file', 'pdf'],
            'command_execution': ['executed', 'ran', 'command', 'terminal'],
            'coding': ['coded', 'implemented', 'built', 'programmed', 'developed'],
            'debugging': ['debugged', 'fixed', 'error', 'bug', 'issue'],
            'git_operation': ['committed', 'pushed', 'pulled', 'cloned', 'merged'],
            'testing': ['tested', 'test', 'spec', 'unit test']
        }
        
        for activity_type, keywords in activity_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                result['activity_type'] = activity_type
                break
        
        # Enhanced temporal expressions
        now = datetime.now()
        
        # Relative time expressions
        relative_patterns = [
            # Last X units
            (r'last\s+(\d+)\s+(minute|hour|day|week|month)s?', self._parse_last_n_units),
            # Past X units
            (r'(?:past|previous)\s+(\d+)\s+(minute|hour|day|week|month)s?', self._parse_last_n_units),
            # X units ago
            (r'(\d+)\s+(minute|hour|day|week|month)s?\s+ago', self._parse_n_units_ago),
            # This morning/afternoon/evening
            (r'this\s+(morning|afternoon|evening|night)', self._parse_time_of_day),
            # Earlier today/this week/this month
            (r'earlier\s+(today|this\s+week|this\s+month)', self._parse_earlier_period),
            # Specific weekdays
            (r'(?:last\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)', self._parse_weekday),
            # Recently (defaults to last 2 hours)
            (r'\b(?:recently|just\s+now|a\s+while\s+ago)\b', self._parse_recently),
            # Ordinal expressions (first, second, last)
            (r'(?:the\s+)?(first|second|third|last)\s+(video|website|page)', self._parse_ordinal)
        ]
        
        for pattern, parser_func in relative_patterns:
            match = re.search(pattern, query_lower)
            if match:
                time_info = parser_func(match, now)
                if time_info:
                    result['time_range'] = time_info
                    break
        
        # Domain-specific temporal expressions
        domain_temporal_patterns = [
            # "last YouTube video I watched"
            (r'last\s+youtube\s+video', lambda m, n: self._get_last_n_hours(n, 24)),
            # "Netflix show I was watching"
            (r'netflix\s+(?:show|movie|series).*(?:watching|watched)', lambda m, n: self._get_last_n_hours(n, 48)),
            # "website I visited earlier"
            (r'website.*(?:visited|browsed).*earlier', lambda m, n: self._get_earlier_today(n))
        ]
        
        for pattern, parser_func in domain_temporal_patterns:
            match = re.search(pattern, query_lower)
            if match:
                time_info = parser_func(match, now)
                if time_info:
                    result['time_range'] = time_info
                    break
        
        # If we found platform but no specific time, set reasonable defaults
        if result.get('platform_filter') and not result.get('time_range') and not existing_time_range:
            # Default to last 24 hours for video platforms, 8 hours for web
            hours = 24 if result['platform_filter'] in ['youtube', 'netflix'] else 8
            result['time_range'] = self._get_last_n_hours(now, hours)
        
        return result if result else None
    
    def _parse_last_n_units(self, match, now: datetime) -> Dict[str, datetime]:
        """Parse 'last N units' expressions."""
        n = int(match.group(1))
        unit = match.group(2)
        
        if unit.startswith('minute'):
            delta = timedelta(minutes=n)
        elif unit.startswith('hour'):
            delta = timedelta(hours=n)
        elif unit.startswith('day'):
            delta = timedelta(days=n)
        elif unit.startswith('week'):
            delta = timedelta(weeks=n)
        elif unit.startswith('month'):
            delta = timedelta(days=n * 30)  # Approximate
        else:
            return {}
        
        return {
            "start": now - delta,
            "end": now
        }
    
    def _parse_n_units_ago(self, match, now: datetime) -> Dict[str, datetime]:
        """Parse 'N units ago' expressions."""
        return self._parse_last_n_units(match, now)
    
    def _parse_time_of_day(self, match, now: datetime) -> Dict[str, datetime]:
        """Parse time-of-day expressions like 'this morning'."""
        period = match.group(1)
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        if period == 'morning':
            return {
                "start": today.replace(hour=6),
                "end": today.replace(hour=12)
            }
        elif period == 'afternoon':
            return {
                "start": today.replace(hour=12),
                "end": today.replace(hour=18)
            }
        elif period == 'evening':
            return {
                "start": today.replace(hour=18),
                "end": today.replace(hour=23, minute=59)
            }
        elif period == 'night':
            return {
                "start": today.replace(hour=22),
                "end": (today + timedelta(days=1)).replace(hour=6)
            }
        
        return {}
    
    def _parse_earlier_period(self, match, now: datetime) -> Dict[str, datetime]:
        """Parse 'earlier today/week/month' expressions."""
        period = match.group(1)
        
        if 'today' in period:
            today = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return {
                "start": today,
                "end": now - timedelta(hours=1)  # Everything except last hour
            }
        elif 'week' in period:
            # Start of this week to 24 hours ago
            days_since_monday = now.weekday()
            start_of_week = now - timedelta(days=days_since_monday)
            start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
            return {
                "start": start_of_week,
                "end": now - timedelta(hours=24)
            }
        elif 'month' in period:
            # Start of this month to 48 hours ago
            start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            return {
                "start": start_of_month,
                "end": now - timedelta(hours=48)
            }
        
        return {}
    
    def _parse_weekday(self, match, now: datetime) -> Dict[str, datetime]:
        """Parse weekday references like 'last monday'."""
        weekday_name = match.group(1)
        weekdays = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        
        target_weekday = weekdays.get(weekday_name)
        if target_weekday is None:
            return {}
        
        # Find the most recent occurrence of this weekday
        days_back = (now.weekday() - target_weekday) % 7
        if days_back == 0 and 'last' in match.group(0):
            days_back = 7  # If it's the same day and they said "last", go back a week
        
        target_date = now - timedelta(days=days_back)
        start_of_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = target_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        return {
            "start": start_of_day,
            "end": end_of_day
        }
    
    def _parse_recently(self, match, now: datetime) -> Dict[str, datetime]:
        """Parse 'recently' to mean last 2 hours."""
        return {
            "start": now - timedelta(hours=2),
            "end": now
        }
    
    def _parse_ordinal(self, match, now: datetime) -> Dict[str, datetime]:
        """Parse ordinal expressions - for now just return recent timeframe."""
        ordinal = match.group(1)
        if ordinal == 'last':
            # Last item - focus on very recent activity
            return {
                "start": now - timedelta(hours=6),
                "end": now
            }
        else:
            # First, second, third - broader timeframe
            return {
                "start": now - timedelta(days=7),
                "end": now
            }
    
    def _get_last_n_hours(self, now: datetime, hours: int) -> Dict[str, datetime]:
        """Helper to get time range for last N hours."""
        return {
            "start": now - timedelta(hours=hours),
            "end": now
        }
    
    def _get_earlier_today(self, now: datetime) -> Dict[str, datetime]:
        """Helper to get earlier today timeframe."""
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return {
            "start": today,
            "end": now - timedelta(hours=2)
        }
    
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
            
            # Extract supported filters
            if filters:
                if "content_type" in filters:
                    search_kwargs["content_type_filter"] = filters["content_type"]
                if "min_confidence" in filters:
                    search_kwargs["min_confidence"] = filters["min_confidence"]
            
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
            # Build comprehensive context with enhanced screenshot metadata
            context = self._build_enhanced_rag_context(query, search_results, max_context_length)
            
            # Create enhanced RAG prompt for intelligent analysis
            rag_prompt = f"""You are analyzing a user's computer activity history to answer their question intelligently. You have access to comprehensive screenshot data, OCR text, and metadata from their digital activities.

{context}

Based on this comprehensive activity data, please provide an intelligent, detailed response that:

1. **Directly answers the question**: {query}
2. **Provides specific details**: Include exact timestamps, commands, file names, URLs, error messages, etc.
3. **Offers context and insights**: Explain what the user was doing and why it matters
4. **Suggests actionable next steps**: If applicable, provide helpful recommendations
5. **Maintains chronological clarity**: Present information in a logical time sequence when relevant

Be specific, practical, and useful. If you don't have complete information, clearly state what you found and what might be missing. Focus on being genuinely helpful rather than generic."""
            
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
    
    def _build_enhanced_rag_context(self, query: str, search_results: List[SearchResult], max_length: int = 2000) -> str:
        """
        Build comprehensive context with enhanced screenshot metadata for LLM analysis.
        
        Args:
            query: Original user query
            search_results: Retrieved search results
            max_length: Maximum context length
            
        Returns:
            Enhanced context string with comprehensive screenshot information
        """
        context_parts = []
        total_length = 0
        
        # Add query context header
        context_header = f"USER QUERY: {query}\n\nRELEVANT ACTIVITY DATA:\n"
        context_parts.append(context_header)
        total_length += len(context_header)
        
        for i, result in enumerate(search_results[:8], 1):  # Use more results for better context
            # Extract comprehensive metadata
            metadata = result.metadata or {}
            timestamp = result.timestamp.strftime("%Y-%m-%d %H:%M:%S") if result.timestamp else "unknown time"
            
            # Build enhanced context entry
            context_entry_parts = [f"\n--- ACTIVITY {i} ({timestamp}) ---"]
            
            # Add basic content
            content = result.content[:400] if result.content else "No content"
            context_entry_parts.append(f"Content: {content}")
            
            # Add platform/domain information
            platform = metadata.get('platform', 'unknown')
            domain = metadata.get('domain', '')
            if platform != 'unknown':
                context_entry_parts.append(f"Platform: {platform}")
            if domain:
                context_entry_parts.append(f"Domain: {domain}")
            
            # Add LLM-enhanced metadata if available
            if metadata.get('llm_enhanced'):
                if metadata.get('key_information'):
                    context_entry_parts.append(f"Key Info: {metadata['key_information']}")
                if metadata.get('context_significance'):
                    context_entry_parts.append(f"Significance: {metadata['context_significance']}")
                if metadata.get('actionable_insights'):
                    context_entry_parts.append(f"Insights: {metadata['actionable_insights']}")
            
            # Add technical details for development queries
            if any(term in query.lower() for term in ['command', 'terminal', 'code', 'error', 'debug']):
                if metadata.get('terminal_command'):
                    context_entry_parts.append(f"Command: {metadata['terminal_command']}")
                if metadata.get('error_type'):
                    context_entry_parts.append(f"Error: {metadata['error_type']}")
                if metadata.get('file_path'):
                    context_entry_parts.append(f"File: {metadata['file_path']}")
            
            # Add URL/title info for web content
            if metadata.get('url'):
                context_entry_parts.append(f"URL: {metadata['url']}")
            if metadata.get('title'):
                context_entry_parts.append(f"Title: {metadata['title']}")
            
            # Add activity type and confidence
            if metadata.get('activity_type'):
                context_entry_parts.append(f"Activity: {metadata['activity_type']}")
            
            context_entry = "\n".join(context_entry_parts)
            
            # Check if adding this entry exceeds the limit
            if total_length + len(context_entry) > max_length:
                # Add partial entry if there's meaningful space
                remaining_space = max_length - total_length - 50  # Leave some buffer
                if remaining_space > 200:
                    partial_entry = context_entry[:remaining_space] + "...\n[truncated]"
                    context_parts.append(partial_entry)
                break
            
            context_parts.append(context_entry)
            total_length += len(context_entry)
        
        # Add query analysis hints
        context_footer = f"\n\n--- ANALYSIS INSTRUCTIONS ---\nPlease analyze the above activity data to answer: {query}\nFocus on specific details, timestamps, and actionable insights. Be comprehensive but concise."
        
        if total_length + len(context_footer) <= max_length:
            context_parts.append(context_footer)
        
        return "\n".join(context_parts)
    
    async def query_activity_timeline(self, query_intent: QueryIntent) -> List[SearchResult]:
        """
        Query the activity timeline using parsed intent.
        
        Args:
            query_intent: Parsed query intent with filters and time range
            
        Returns:
            List of SearchResult objects from activity timeline
        """
        try:
            filters = query_intent.filters
            time_range = query_intent.time_range
            
            # Check if this is a temporal query that should use activity timeline
            if (filters.get('platform') or filters.get('domain') or filters.get('activity_type') 
                or any(term in query_intent.original_query.lower() for term in ['last', 'recent', 'watched', 'visited'])):
                
                results = []
                
                # Query by platform (normalize case for known platforms)
                if filters.get('platform'):
                    platform = filters['platform']
                    # Normalize platform names to match database format
                    platform_map = {
                        'youtube': 'YouTube',
                        'netflix': 'Netflix',
                        'twitter': 'Twitter',
                        'facebook': 'Facebook',
                        'instagram': 'Instagram'
                    }
                    normalized_platform = platform_map.get(platform.lower(), platform)
                    
                    platform_results = self.metadata_db.get_last_activity_by_platform(
                        normalized_platform, limit=10
                    )
                    results.extend(self._convert_activity_to_search_results(platform_results))
                
                # Query by domain
                elif filters.get('domain'):
                    domain_results = self.metadata_db.get_last_activity_by_domain(
                        filters['domain'], limit=10
                    )
                    results.extend(self._convert_activity_to_search_results(domain_results))
                
                # Query by time range
                elif time_range:
                    time_results = self.metadata_db.get_activities_by_time_range(
                        time_range['start'], time_range['end'], 
                        platform=filters.get('platform'), limit=20
                    )
                    results.extend(self._convert_activity_to_search_results(time_results))
                
                # Search activity timeline with FTS
                if query_intent.search_terms:
                    search_query = ' '.join(query_intent.search_terms)
                    fts_results = self.metadata_db.search_activity_timeline(
                        search_query, platform=filters.get('platform'), limit=15
                    )
                    results.extend(self._convert_activity_to_search_results(fts_results))
                
                # Remove duplicates and sort by time
                seen_ids = set()
                unique_results = []
                for result in results:
                    if result.content_id not in seen_ids:
                        seen_ids.add(result.content_id)
                        unique_results.append(result)
                
                # Sort by timestamp (most recent first)
                unique_results.sort(key=lambda x: x.timestamp, reverse=True)
                
                return unique_results[:10]  # Return top 10
                
        except Exception as e:
            self.logger.error(f"Activity timeline query failed: {e}")
        
        return []
    
    def _convert_activity_to_search_results(self, activity_results: List[Dict[str, Any]]) -> List[SearchResult]:
        """Convert activity timeline results to SearchResult objects."""
        search_results = []
        
        for activity in activity_results:
            # Build content description
            content_parts = []
            
            if activity.get('title'):
                content_parts.append(f"Title: {activity['title']}")
            
            if activity.get('platform'):
                content_parts.append(f"Platform: {activity['platform']}")
            
            if activity.get('url'):
                content_parts.append(f"URL: {activity['url']}")
            
            if activity.get('activity_type'):
                content_parts.append(f"Activity: {activity['activity_type']}")
            
            # Add metadata content
            if activity.get('metadata') and isinstance(activity['metadata'], dict):
                for key, value in activity['metadata'].items():
                    if value and str(value).strip():
                        content_parts.append(f"{key}: {value}")
            
            content = '\n'.join(content_parts) if content_parts else activity.get('title', 'Activity')
            
            # Parse timestamp
            timestamp = activity.get('session_start') or activity.get('screenshot_timestamp')
            if isinstance(timestamp, str):
                try:
                    from datetime import datetime
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except ValueError:
                    timestamp = datetime.now()
            elif not isinstance(timestamp, datetime):
                timestamp = datetime.now()
            
            search_result = SearchResult(
                content_id=f"activity_{activity.get('id', 'unknown')}",
                content=content,
                similarity_score=0.9,  # High relevance for timeline queries
                timestamp=timestamp,
                metadata={
                    'activity_id': activity.get('id'),
                    'screenshot_id': activity.get('screenshot_id'),
                    'platform': activity.get('platform'),
                    'domain': activity.get('domain'),
                    'activity_type': activity.get('activity_type'),
                    'title': activity.get('title'),
                    'url': activity.get('url'),
                    'source_type': 'activity_timeline',
                    **activity.get('metadata', {})
                },
                source_type="activity_timeline"
            )
            
            search_results.append(search_result)
        
        return search_results
    
    async def handle_specialized_query(self, query: str, query_intent: QueryIntent, search_results: List[SearchResult]) -> Optional[str]:
        """
        Handle specialized queries with domain-specific responses.
        
        Args:
            query: Original query string
            query_intent: Parsed query intent
            search_results: Search results from timeline and semantic search
            
        Returns:
            Specialized response string or None if not applicable
        """
        query_lower = query.lower()
        filters = query_intent.filters
        
        # YouTube video queries
        if (filters.get('platform') == 'youtube' or 
            any(term in query_lower for term in ['youtube', 'video', 'watched'])):
            return await self._handle_youtube_query(query, search_results)
        
        # Netflix content queries
        elif (filters.get('platform') == 'netflix' or 
              any(term in query_lower for term in ['netflix', 'movie', 'show', 'episode'])):
            return await self._handle_netflix_query(query, search_results)
        
        # Website/browsing queries
        elif (filters.get('platform') == 'browser' or 
              any(term in query_lower for term in ['website', 'visited', 'browsed', 'webpage'])):
            return await self._handle_website_query(query, search_results)
        
        # Social media queries
        elif (filters.get('platform') == 'social_media' or 
              any(term in query_lower for term in ['twitter', 'facebook', 'instagram', 'social'])):
            return await self._handle_social_media_query(query, search_results)
        
        # Terminal/command queries
        elif (filters.get('platform') == 'terminal' or query_intent.intent_type == 'terminal' or
              any(term in query_lower for term in ['terminal', 'command', 'cmd', 'shell', 'executed'])):
            return await self._handle_terminal_query(query, search_results)
        
        # Development/coding queries
        elif (filters.get('platform') == 'development' or query_intent.intent_type == 'development' or
              any(term in query_lower for term in ['implement', 'coded', 'built', 'feature', 'function'])):
            return await self._handle_development_query(query, search_results)
        
        # Error analysis queries
        elif (query_intent.intent_type == 'error_analysis' or
              any(term in query_lower for term in ['wrong', 'error', 'failed', 'bug', 'debug'])):
            return await self._handle_error_analysis_query(query, search_results)
        
        # Git/version control queries
        elif (query_intent.intent_type == 'git' or
              any(term in query_lower for term in ['git', 'clone', 'commit', 'push', 'pull', 'repo'])):
            return await self._handle_git_query(query, search_results)
        
        # Temporal/recent activity queries
        elif any(term in query_lower for term in ['last', 'recent', 'lately', 'earlier']):
            return await self._handle_temporal_query(query, search_results)
        
        return None
    
    async def _handle_youtube_query(self, query: str, search_results: List[SearchResult]) -> str:
        """Handle YouTube-specific queries."""
        if not search_results:
            return "I couldn't find any YouTube activity in your recent history."
        
        query_lower = query.lower()
        
        # Filter for YouTube results (case-insensitive platform check)
        youtube_results = [r for r in search_results if 
                          str(r.metadata.get('platform', '')).lower() == 'youtube' or 
                          'youtube' in r.content.lower()]
        
        if not youtube_results:
            return "I couldn't find any YouTube videos in your recent activity."
        
        # Check for specific query types
        if 'last' in query_lower and 'video' in query_lower:
            # "What's the last YouTube video I watched?"
            latest_video = youtube_results[0]  # Already sorted by time
            
            response_parts = []
            
            # Extract video information
            video_info = self._extract_video_info_from_result(latest_video)
            
            if video_info.get('title'):
                response_parts.append(f"The last YouTube video you watched was: **{video_info['title']}**")
            else:
                response_parts.append("The last YouTube video you watched:")
            
            if video_info.get('channel'):
                response_parts.append(f"Channel: {video_info['channel']}")
            
            if video_info.get('duration'):
                response_parts.append(f"Duration: {video_info['duration']}")
            
            if video_info.get('view_count'):
                response_parts.append(f"Views: {video_info['view_count']}")
            
            # Add timestamp
            timestamp = latest_video.timestamp
            if timestamp:
                time_str = timestamp.strftime("%Y-%m-%d at %H:%M")
                response_parts.append(f"Watched on: {time_str}")
            
            return "\n".join(response_parts)
        
        elif any(term in query_lower for term in ['videos', 'list', 'all']):
            # "Show me YouTube videos I watched"
            response_parts = [f"Here are {len(youtube_results)} YouTube videos from your recent activity:"]
            
            for i, result in enumerate(youtube_results[:5], 1):
                video_info = self._extract_video_info_from_result(result)
                title = video_info.get('title', 'Unknown Video')
                channel = video_info.get('channel', '')
                timestamp = result.timestamp.strftime("%m/%d %H:%M") if result.timestamp else "Unknown time"
                
                entry = f"{i}. **{title}**"
                if channel:
                    entry += f" - {channel}"
                entry += f" ({timestamp})"
                
                response_parts.append(entry)
            
            if len(youtube_results) > 5:
                response_parts.append(f"... and {len(youtube_results) - 5} more videos")
            
            return "\n".join(response_parts)
        
        else:
            # General YouTube activity summary
            return self._generate_general_activity_summary(youtube_results, "YouTube videos")
    
    async def _handle_netflix_query(self, query: str, search_results: List[SearchResult]) -> str:
        """Handle Netflix-specific queries."""
        if not search_results:
            return "I couldn't find any Netflix activity in your recent history."
        
        query_lower = query.lower()
        
        # Filter for Netflix results
        netflix_results = [r for r in search_results if 
                          r.metadata.get('platform') == 'netflix' or 
                          'netflix' in r.content.lower()]
        
        if not netflix_results:
            return "I couldn't find any Netflix content in your recent activity."
        
        if 'last' in query_lower or 'recent' in query_lower:
            # "What's the last Netflix show I watched?"
            latest_content = netflix_results[0]
            
            response_parts = []
            video_info = self._extract_video_info_from_result(latest_content)
            
            if video_info.get('title'):
                content_type = "movie" if any(term in query_lower for term in ['movie', 'film']) else "show"
                response_parts.append(f"The last Netflix {content_type} you watched was: **{video_info['title']}**")
            
            if video_info.get('season') and video_info.get('episode'):
                response_parts.append(f"Season {video_info['season']}, Episode {video_info['episode']}")
            elif video_info.get('season'):
                response_parts.append(f"Season {video_info['season']}")
            
            if video_info.get('genre'):
                response_parts.append(f"Genre: {video_info['genre']}")
            
            if video_info.get('rating'):
                response_parts.append(f"Rating: {video_info['rating']}")
            
            timestamp = latest_content.timestamp
            if timestamp:
                time_str = timestamp.strftime("%Y-%m-%d at %H:%M")
                response_parts.append(f"Watched on: {time_str}")
            
            return "\n".join(response_parts)
        
        else:
            return self._generate_general_activity_summary(netflix_results, "Netflix content")
    
    async def _handle_website_query(self, query: str, search_results: List[SearchResult]) -> str:
        """Handle website/browsing queries."""
        if not search_results:
            return "I couldn't find any website activity in your recent history."
        
        query_lower = query.lower()
        
        # Filter for website results
        website_results = [r for r in search_results if 
                          r.metadata.get('platform') == 'browser' or 
                          r.metadata.get('activity_type') == 'webpage_visit' or
                          any(term in r.content.lower() for term in ['url', 'website', 'domain'])]
        
        if not website_results:
            return "I couldn't find any website visits in your recent activity."
        
        if 'last' in query_lower or 'recent' in query_lower:
            # "What's the last website I visited?"
            latest_site = website_results[0]
            
            response_parts = []
            
            # Extract website information
            title = latest_site.metadata.get('title', '')
            url = latest_site.metadata.get('url', '')
            domain = latest_site.metadata.get('domain', '')
            
            if title:
                response_parts.append(f"The last website you visited was: **{title}**")
            elif domain:
                response_parts.append(f"The last website you visited was: **{domain}**")
            else:
                response_parts.append("The last website you visited:")
            
            if url:
                response_parts.append(f"URL: {url}")
            elif domain:
                response_parts.append(f"Domain: {domain}")
            
            timestamp = latest_site.timestamp
            if timestamp:
                time_str = timestamp.strftime("%Y-%m-%d at %H:%M")
                response_parts.append(f"Visited on: {time_str}")
            
            return "\n".join(response_parts)
        
        else:
            return self._generate_general_activity_summary(website_results, "websites")
    
    async def _handle_social_media_query(self, query: str, search_results: List[SearchResult]) -> str:
        """Handle social media queries."""
        if not search_results:
            return "I couldn't find any social media activity in your recent history."
        
        social_results = [r for r in search_results if 
                         r.metadata.get('platform') == 'social_media' or
                         any(term in r.content.lower() for term in ['twitter', 'facebook', 'instagram', 'linkedin'])]
        
        if not social_results:
            return "I couldn't find any social media activity in your recent history."
        
        return self._generate_general_activity_summary(social_results, "social media activity")
    
    async def _handle_temporal_query(self, query: str, search_results: List[SearchResult]) -> str:
        """Handle temporal queries like 'what did I do recently'."""
        if not search_results:
            return "I couldn't find any recent activity in your history."
        
        query_lower = query.lower()
        
        # Group results by platform/type
        platform_counts = {}
        recent_activities = []
        
        for result in search_results[:10]:  # Focus on most recent
            platform = result.metadata.get('platform', 'unknown')
            platform_counts[platform] = platform_counts.get(platform, 0) + 1
            
            activity_summary = self._summarize_activity_result(result)
            recent_activities.append(activity_summary)
        
        response_parts = []
        
        if 'recently' in query_lower or 'lately' in query_lower:
            response_parts.append("Here's what you've been doing recently:")
        else:
            response_parts.append("Recent activity summary:")
        
        # Show platform breakdown
        if platform_counts:
            platform_summary = []
            for platform, count in sorted(platform_counts.items(), key=lambda x: x[1], reverse=True):
                if platform != 'unknown':
                    platform_summary.append(f"{platform}: {count} activities")
            
            if platform_summary:
                response_parts.append(f"\n**Activity breakdown:** {', '.join(platform_summary)}")
        
        # Show recent activities
        response_parts.append("\n**Recent activities:**")
        for i, activity in enumerate(recent_activities[:5], 1):
            response_parts.append(f"{i}. {activity}")
        
        return "\n".join(response_parts)
    
    async def _handle_terminal_query(self, query: str, search_results: List[SearchResult]) -> str:
        """Handle terminal/command queries."""
        if not search_results:
            return "I couldn't find any terminal commands in your recent history."
        
        query_lower = query.lower()
        
        # Filter for terminal/command results
        terminal_results = [r for r in search_results if 
                           r.metadata.get('platform') == 'terminal' or 
                           r.metadata.get('content_type') == 'terminal' or
                           any(term in r.content.lower() for term in ['command', 'terminal', '$', '>', 'bash', 'zsh'])]
        
        if not terminal_results:
            return "I couldn't find any terminal commands in your recent activity."
        
        # Check for specific query types
        if 'last' in query_lower and 'command' in query_lower:
            # "What was the last command I ran?"
            latest_command = terminal_results[0]  # Already sorted by time
            
            response_parts = []
            
            # Extract command from metadata or content
            command = latest_command.metadata.get('terminal_command') or latest_command.metadata.get('command')
            if not command:
                # Try to extract from content
                import re
                content = latest_command.content
                # Look for command patterns
                cmd_patterns = [r'\$ (.+)', r'> (.+)', r'% (.+)', r'# (.+)']
                for pattern in cmd_patterns:
                    match = re.search(pattern, content)
                    if match:
                        command = match.group(1).strip()
                        break
            
            if command:
                response_parts.append(f"The last command you ran was: **{command}**")
            else:
                response_parts.append("The last terminal activity:")
            
            timestamp = latest_command.timestamp
            if timestamp:
                time_str = timestamp.strftime("%Y-%m-%d at %H:%M")
                response_parts.append(f"Executed on: {time_str}")
            
            # Add context if available
            if latest_command.metadata.get('working_directory'):
                response_parts.append(f"Directory: {latest_command.metadata['working_directory']}")
            
            # Add output if available
            if latest_command.metadata.get('command_output'):
                output = latest_command.metadata['command_output'][:200]
                response_parts.append(f"Output: {output}...")
            
            return "\n".join(response_parts)
        
        else:
            # General terminal activity summary
            response_parts = [f"Found {len(terminal_results)} terminal commands:"]
            
            for i, result in enumerate(terminal_results[:5], 1):
                timestamp_str = result.timestamp.strftime("%H:%M") if result.timestamp else "unknown time"
                command = result.metadata.get('terminal_command') or "terminal activity"
                response_parts.append(f"{i}. {timestamp_str}: {command}")
            
            return "\n".join(response_parts)
    
    async def _handle_development_query(self, query: str, search_results: List[SearchResult]) -> str:
        """Handle development/coding queries."""
        if not search_results:
            return "I couldn't find any development activity in your recent history."
        
        query_lower = query.lower()
        
        # Filter for development results
        dev_results = [r for r in search_results if 
                      r.metadata.get('platform') == 'development' or
                      r.metadata.get('content_type') in ['code', 'editor'] or
                      any(term in r.content.lower() for term in ['code', 'function', 'class', 'def', 'import', 'git'])]
        
        if not dev_results:
            return "I couldn't find any development activity in your recent history."
        
        # Check for specific query types
        if any(term in query_lower for term in ['implement', 'feature', 'function']):
            response_parts = [f"Recent development activities ({len(dev_results)} found):"]
            
            for i, result in enumerate(dev_results[:5], 1):
                timestamp_str = result.timestamp.strftime("%Y-%m-%d %H:%M") if result.timestamp else "unknown time"
                
                # Extract development info
                activity_desc = result.metadata.get('activity_description', result.content[:100])
                file_info = result.metadata.get('file_path', '')
                
                response_parts.append(f"\n{i}. **{timestamp_str}**")
                response_parts.append(f"   Activity: {activity_desc}")
                if file_info:
                    response_parts.append(f"   File: {file_info}")
                
                # Add LLM insights if available
                if result.metadata.get('key_information'):
                    response_parts.append(f"   Details: {result.metadata['key_information']}")
            
            return "\n".join(response_parts)
        
        else:
            return self._generate_general_activity_summary(dev_results, "development activities")
    
    async def _handle_error_analysis_query(self, query: str, search_results: List[SearchResult]) -> str:
        """Handle error analysis queries."""
        if not search_results:
            return "I couldn't find any error information in your recent history."
        
        query_lower = query.lower()
        
        # Filter for error-related results
        error_results = [r for r in search_results if 
                        r.metadata.get('error_type') or
                        any(term in r.content.lower() for term in ['error', 'exception', 'failed', 'traceback', 'warning'])]
        
        if not error_results:
            return "I couldn't find any errors in your recent activity."
        
        # Check for time-specific queries
        if any(term in query_lower for term in ['last hour', 'hour', 'recently']):
            response_parts = ["**Error Analysis - Last Hour:**"]
            
            for i, result in enumerate(error_results[:3], 1):
                timestamp_str = result.timestamp.strftime("%H:%M") if result.timestamp else "unknown time"
                
                error_type = result.metadata.get('error_type', 'Unknown error')
                error_desc = result.metadata.get('error_description', result.content[:150])
                
                response_parts.append(f"\n{i}. **{timestamp_str}** - {error_type}")
                response_parts.append(f"   Description: {error_desc}")
                
                # Add context from LLM analysis
                if result.metadata.get('context_significance'):
                    response_parts.append(f"   Context: {result.metadata['context_significance']}")
                
                # Add actionable insights
                if result.metadata.get('actionable_insights'):
                    response_parts.append(f"   Suggestion: {result.metadata['actionable_insights']}")
            
            return "\n".join(response_parts)
        
        else:
            return self._generate_general_activity_summary(error_results, "errors and issues")
    
    async def _handle_git_query(self, query: str, search_results: List[SearchResult]) -> str:
        """Handle git/version control queries."""
        if not search_results:
            return "I couldn't find any git activity in your recent history."
        
        query_lower = query.lower()
        
        # Filter for git results
        git_results = [r for r in search_results if 
                      r.metadata.get('git_operation') or
                      any(term in r.content.lower() for term in ['git', 'clone', 'commit', 'push', 'pull', 'merge', 'branch'])]
        
        if not git_results:
            return "I couldn't find any git activity in your recent history."
        
        # Check for specific operations
        if 'clone' in query_lower:
            clone_results = [r for r in git_results if 'clone' in r.content.lower()]
            if clone_results:
                latest_clone = clone_results[0]
                timestamp_str = latest_clone.timestamp.strftime("%Y-%m-%d at %H:%M") if latest_clone.timestamp else "unknown time"
                
                # Extract repository info
                repo_url = latest_clone.metadata.get('repository_url', '')
                if not repo_url:
                    # Try to extract from content
                    import re
                    content = latest_clone.content
                    url_match = re.search(r'git clone\s+(\S+)', content)
                    if url_match:
                        repo_url = url_match.group(1)
                
                response_parts = [f"Last repository cloned: **{repo_url}**" if repo_url else "Last git clone operation:"]
                response_parts.append(f"Cloned on: {timestamp_str}")
                
                return "\n".join(response_parts)
        
        else:
            # General git activity summary
            response_parts = [f"Recent git activities ({len(git_results)} found):"]
            
            for i, result in enumerate(git_results[:5], 1):
                timestamp_str = result.timestamp.strftime("%H:%M") if result.timestamp else "unknown time"
                git_op = result.metadata.get('git_operation', 'git activity')
                response_parts.append(f"{i}. {timestamp_str}: {git_op}")
            
            return "\n".join(response_parts)
    
    def _extract_video_info_from_result(self, result: SearchResult) -> Dict[str, str]:
        """Extract video information from search result."""
        info = {}
        
        # Try to extract from metadata first
        metadata = result.metadata
        info['title'] = metadata.get('title', '')
        info['channel'] = metadata.get('channel_name', '')
        info['duration'] = metadata.get('duration', '')
        info['view_count'] = metadata.get('view_count', '')
        info['season'] = metadata.get('season', '')
        info['episode'] = metadata.get('episode', '')
        info['genre'] = metadata.get('genre', '')
        info['rating'] = metadata.get('rating', '')
        
        # Try to extract from content if not in metadata
        content = result.content
        
        if not info['title'] and 'Title:' in content:
            import re
            title_match = re.search(r'Title:\s*(.+)', content)
            if title_match:
                info['title'] = title_match.group(1).strip()
        
        if not info['channel'] and 'Channel:' in content:
            import re
            channel_match = re.search(r'Channel:\s*(.+)', content)
            if channel_match:
                info['channel'] = channel_match.group(1).strip()
        
        return info
    
    def _generate_general_activity_summary(self, results: List[SearchResult], activity_type: str) -> str:
        """Generate a general summary for activity results."""
        if not results:
            return f"No {activity_type} found in your recent activity."
        
        response_parts = [f"Found {len(results)} {activity_type} in your recent activity:"]
        
        for i, result in enumerate(results[:3], 1):
            summary = self._summarize_activity_result(result)
            response_parts.append(f"{i}. {summary}")
        
        if len(results) > 3:
            response_parts.append(f"... and {len(results) - 3} more items")
        
        return "\n".join(response_parts)
    
    def _summarize_activity_result(self, result: SearchResult) -> str:
        """Create a brief summary of an activity result."""
        title = result.metadata.get('title', '')
        platform = result.metadata.get('platform', '')
        timestamp = result.timestamp
        
        time_str = timestamp.strftime("%m/%d %H:%M") if timestamp else "Unknown time"
        
        if title:
            summary = f"**{title}**"
            if platform:
                summary += f" ({platform})"
            summary += f" - {time_str}"
        elif platform:
            summary = f"{platform} activity - {time_str}"
        else:
            summary = f"Activity - {time_str}"
        
        return summary
    
    @log_performance
    async def process_natural_language_query(self, query: str) -> MemoryResponse:
        """
        Process a natural language query end-to-end with FastMCP optimization.
        
        Args:
            query: Natural language query string
            
        Returns:
            MemoryResponse with search results and generated response
        """
        try:
            # Get FastMCP processor for query optimization
            fastmcp = self._get_fastmcp()
            
            # Get query optimization suggestions from FastMCP
            query_context = {
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'system': 'memory_system'
            }
            
            optimization_hints = await fastmcp.optimize_query_processing(query, query_context)
            
            # Parse the query to understand intent
            query_intent = self.parse_natural_language_query(query)
            
            # Apply FastMCP optimizations
            if optimization_hints.get('parallel_processing'):
                # Use parallel processing for complex queries
                activity_task = asyncio.create_task(self.query_activity_timeline(query_intent))
                search_task = asyncio.create_task(self._get_optimized_search_results(query_intent, optimization_hints))
                
                activity_results, search_results = await asyncio.gather(activity_task, search_task)
            else:
                # Sequential processing for simple queries
                activity_results = await self.query_activity_timeline(query_intent)
                search_results = await self._get_optimized_search_results(query_intent, optimization_hints)
            
            # Additional search logic moved to _get_optimized_search_results method
            
            # Combine activity timeline results with regular search results
            # Activity results get priority (higher relevance for temporal queries)
            combined_results = activity_results + search_results
            
            # Remove duplicates and limit results
            seen_content = set()
            final_results = []
            for result in combined_results:
                content_hash = hash(result.content[:100])  # Hash first 100 chars
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    final_results.append(result)
                if len(final_results) >= 15:  # Limit to 15 total results
                    break
            
            search_results = final_results
            
            # Try specialized query handling first
            specialized_response = await self.handle_specialized_query(query, query_intent, search_results)
            
            if specialized_response:
                # Use specialized response
                response = specialized_response
                generated_by = "specialized"
                confidence = 0.9
            elif query_intent.intent_type in ["summarize", "analyze", "compare"] and search_results:
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
    
    def _get_fastmcp(self):
        """Get or create FastMCP instance (singleton pattern)."""
        if self.fastmcp is None:
            self.fastmcp = get_fastmcp()
            self.logger.info("FastMCP integrated with memory system for query optimization")
        return self.fastmcp
    
    async def _get_optimized_search_results(self, query_intent, optimization_hints) -> List[SearchResult]:
        """Get search results with FastMCP optimizations applied."""
        try:
            # Apply cache strategy from FastMCP
            cache_strategy = optimization_hints.get('cache_strategy', 'standard')
            
            if query_intent.intent_type in ["summarize", "analyze", "compare"]:
                # For complex queries, use hybrid search
                search_results = await self.hybrid_search(
                    " ".join(query_intent.search_terms),
                    n_results=15,
                    filters=query_intent.filters
                )
            elif cache_strategy == 'temporal' and query_intent.filters.get('platform'):
                # Optimized search for temporal platform-specific queries
                search_results = await self.semantic_search(
                    " ".join(query_intent.search_terms),
                    n_results=8,  # Smaller result set for faster response
                    filters=query_intent.filters
                )
            else:
                # Default semantic search
                search_results = await self.semantic_search(
                    " ".join(query_intent.search_terms),
                    n_results=10,
                    filters=query_intent.filters
                )
            
            # Log performance hints from FastMCP
            if optimization_hints.get('performance_hints'):
                self.logger.debug(f"FastMCP performance hints: {optimization_hints['performance_hints']}")
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Optimized search failed: {e}")
            # Fallback to basic semantic search
            return await self.semantic_search(
                " ".join(query_intent.search_terms),
                n_results=10,
                filters=query_intent.filters
            )