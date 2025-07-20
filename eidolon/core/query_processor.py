"""
Advanced Query Processing Engine for Eidolon AI Personal Assistant

Provides sophisticated query processing capabilities including natural language parsing,
multi-condition filtering, temporal queries, aggregation, and cross-reference support.
"""

import os
import sqlite3
import json
import re
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
import asyncio

from ..utils.logging import get_component_logger, log_performance, log_exceptions
from ..utils.config import get_config
from ..storage.metadata_db import MetadataDatabase
from ..storage.vector_db import VectorDatabase
from .analytics import AnalyticsEngine


@dataclass
class QueryCondition:
    """Represents a single query condition."""
    field: str
    operator: str  # 'eq', 'ne', 'gt', 'lt', 'gte', 'lte', 'contains', 'in', 'between'
    value: Any
    table: str = 'screenshots'


@dataclass
class TemporalQuery:
    """Represents temporal query parameters."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    time_window: Optional[str] = None  # 'today', 'yesterday', 'last_week', 'last_month'
    recurring_pattern: Optional[str] = None  # 'daily', 'weekly', 'monthly'


@dataclass
class AggregationQuery:
    """Represents aggregation parameters."""
    group_by: List[str]
    functions: List[str]  # 'count', 'sum', 'avg', 'min', 'max'
    having_conditions: List[QueryCondition] = None


@dataclass
class QueryResult:
    """Represents query execution results."""
    data: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    execution_time_ms: float
    total_count: int
    aggregations: Optional[Dict[str, Any]] = None
    related_results: Optional[List[Dict[str, Any]]] = None


class NaturalLanguageParser:
    """Parses natural language queries into structured query components."""
    
    def __init__(self):
        self.logger = get_component_logger("query.nlp")
        
        # Intent patterns
        self.intent_patterns = {
            'search': [
                r'find|search|look for|show me|get',
                r'what|where|when|how',
                r'screenshots|images|content'
            ],
            'timeline': [
                r'timeline|chronology|sequence|order',
                r'when did|what happened|progress',
                r'project|development|work'
            ],
            'analytics': [
                r'analyze|analysis|insights|patterns',
                r'productivity|time|usage|habits',
                r'statistics|metrics|performance'
            ],
            'filter': [
                r'from|to|between|during|since',
                r'application|app|program|software',
                r'containing|with|including'
            ]
        }
        
        # Temporal patterns
        self.temporal_patterns = {
            'today': r'today|now',
            'yesterday': r'yesterday',
            'last_week': r'last week|past week|previous week',
            'last_month': r'last month|past month|previous month',
            'this_week': r'this week|current week',
            'this_month': r'this month|current month'
        }
        
        # Application patterns
        self.app_patterns = {
            'vscode': r'vscode|visual studio code|code editor',
            'chrome': r'chrome|browser|web',
            'terminal': r'terminal|command line|shell',
            'slack': r'slack|chat|messaging',
            'email': r'email|mail|outlook'
        }
        
    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse natural language query into structured components."""
        query_lower = query.lower()
        
        parsed = {
            'intent': self._detect_intent(query_lower),
            'entities': self._extract_entities(query_lower),
            'temporal': self._extract_temporal(query_lower),
            'filters': self._extract_filters(query_lower),
            'aggregations': self._extract_aggregations(query_lower),
            'original_query': query
        }
        
        return parsed
    
    def _detect_intent(self, query: str) -> str:
        """Detect the primary intent of the query."""
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query):
                    score += 1
            intent_scores[intent] = score
        
        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        
        return 'search'  # Default intent
    
    def _extract_entities(self, query: str) -> List[Dict[str, str]]:
        """Extract named entities from the query."""
        entities = []
        
        # Extract applications
        for app, pattern in self.app_patterns.items():
            if re.search(pattern, query):
                entities.append({
                    'type': 'application',
                    'value': app,
                    'original': re.search(pattern, query).group()
                })
        
        # Extract file extensions
        file_ext_pattern = r'\.(py|js|ts|html|css|md|txt|pdf|doc)'
        matches = re.finditer(file_ext_pattern, query)
        for match in matches:
            entities.append({
                'type': 'file_extension',
                'value': match.group(1),
                'original': match.group()
            })
        
        # Extract keywords in quotes
        quoted_pattern = r'"([^"]+)"'
        matches = re.finditer(quoted_pattern, query)
        for match in matches:
            entities.append({
                'type': 'quoted_text',
                'value': match.group(1),
                'original': match.group()
            })
        
        return entities
    
    def _extract_temporal(self, query: str) -> Optional[TemporalQuery]:
        """Extract temporal information from the query."""
        temporal = TemporalQuery()
        
        # Check for relative time patterns
        for time_window, pattern in self.temporal_patterns.items():
            if re.search(pattern, query):
                temporal.time_window = time_window
                break
        
        # Extract specific dates
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
            r'(\d{2}/\d{2}/\d{4})',  # MM/DD/YYYY
            r'(\d{1,2}/\d{1,2})'     # MM/DD
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, query)
            if matches:
                # Parse the first date found
                try:
                    date_str = matches[0]
                    if '-' in date_str:
                        temporal.start_date = datetime.strptime(date_str, '%Y-%m-%d')
                    elif len(date_str.split('/')) == 3:
                        temporal.start_date = datetime.strptime(date_str, '%m/%d/%Y')
                except ValueError:
                    pass
        
        # Check for time ranges
        range_pattern = r'from\s+(.+?)\s+to\s+(.+?)(?:\s|$)'
        range_match = re.search(range_pattern, query)
        if range_match:
            # Parse date range - simplified for now
            temporal.start_date = self._parse_flexible_date(range_match.group(1))
            temporal.end_date = self._parse_flexible_date(range_match.group(2))
        
        return temporal if any([temporal.start_date, temporal.end_date, temporal.time_window]) else None
    
    def _extract_filters(self, query: str) -> List[QueryCondition]:
        """Extract filter conditions from the query."""
        conditions = []
        
        # Application filters
        app_filter_pattern = r'(?:in|from|using)\s+(\w+)'
        matches = re.finditer(app_filter_pattern, query)
        for match in matches:
            app_name = match.group(1)
            conditions.append(QueryCondition(
                field='application',
                operator='contains',
                value=app_name,
                table='screenshots'
            ))
        
        # Content filters
        content_patterns = [
            (r'containing\s+"([^"]+)"', 'contains'),
            (r'with\s+"([^"]+)"', 'contains'),
            (r'including\s+"([^"]+)"', 'contains')
        ]
        
        for pattern, operator in content_patterns:
            matches = re.finditer(pattern, query)
            for match in matches:
                conditions.append(QueryCondition(
                    field='ocr_text',
                    operator=operator,
                    value=match.group(1),
                    table='screenshots'
                ))
        
        return conditions
    
    def _extract_aggregations(self, query: str) -> Optional[AggregationQuery]:
        """Extract aggregation requirements from the query."""
        if not any(word in query for word in ['count', 'total', 'average', 'sum', 'statistics']):
            return None
        
        group_by = []
        functions = []
        
        # Detect grouping
        if 'by application' in query or 'per app' in query:
            group_by.append('application')
        if 'by day' in query or 'daily' in query:
            group_by.append('date')
        if 'by hour' in query or 'hourly' in query:
            group_by.append('hour')
        
        # Detect functions
        if 'count' in query or 'total' in query or 'how many' in query:
            functions.append('count')
        if 'average' in query or 'avg' in query:
            functions.append('avg')
        if 'sum' in query:
            functions.append('sum')
        
        if group_by or functions:
            return AggregationQuery(group_by=group_by, functions=functions)
        
        return None
    
    def _parse_flexible_date(self, date_str: str) -> Optional[datetime]:
        """Parse flexible date strings."""
        date_str = date_str.strip().lower()
        
        if date_str in ['today', 'now']:
            return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        elif date_str == 'yesterday':
            return datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        
        # Try standard formats
        formats = ['%Y-%m-%d', '%m/%d/%Y', '%m/%d']
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None


class QueryProcessor:
    """Advanced query processor with multi-source data integration."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_component_logger("query_processor")
        
        # Initialize data sources
        self.metadata_db = MetadataDatabase()
        self.vector_db = VectorDatabase()
        self.analytics_engine = AnalyticsEngine()
        self.nlp_parser = NaturalLanguageParser()
        
        # Query cache
        self.query_cache = {}
        self.cache_ttl = timedelta(minutes=5)
        
        self.logger.info("Query processor initialized")
    
    @log_performance
    async def process_query(
        self,
        query: Union[str, Dict[str, Any]],
        limit: int = 100,
        offset: int = 0,
        include_related: bool = True,
        use_cache: bool = True
    ) -> QueryResult:
        """
        Process a query and return results.
        
        Args:
            query: Natural language query string or structured query dict
            limit: Maximum number of results to return
            offset: Number of results to skip
            include_related: Whether to include related content
            use_cache: Whether to use query cache
            
        Returns:
            QueryResult with data and metadata
        """
        start_time = datetime.now()
        
        # Parse query if it's a string
        if isinstance(query, str):
            parsed_query = self.nlp_parser.parse_query(query)
        else:
            parsed_query = query
        
        # Check cache first
        cache_key = self._generate_cache_key(parsed_query, limit, offset)
        if use_cache and cache_key in self.query_cache:
            cached_result, cache_time = self.query_cache[cache_key]
            if datetime.now() - cache_time < self.cache_ttl:
                return cached_result
        
        # Execute query based on intent
        intent = parsed_query.get('intent', 'search')
        
        try:
            if intent == 'search':
                result = await self._execute_search_query(parsed_query, limit, offset)
            elif intent == 'timeline':
                result = await self._execute_timeline_query(parsed_query, limit, offset)
            elif intent == 'analytics':
                result = await self._execute_analytics_query(parsed_query, limit, offset)
            else:
                result = await self._execute_search_query(parsed_query, limit, offset)
            
            # Add related content if requested
            if include_related and result.data:
                result.related_results = await self._find_related_content(result.data[:5])
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            result.execution_time_ms = execution_time
            
            # Cache the result
            if use_cache:
                self.query_cache[cache_key] = (result, datetime.now())
            
            self.logger.info(f"Query processed in {execution_time:.2f}ms, returned {len(result.data)} results")
            return result
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            return QueryResult(
                data=[],
                metadata={'error': str(e)},
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                total_count=0
            )
    
    async def _execute_search_query(
        self, 
        parsed_query: Dict[str, Any], 
        limit: int, 
        offset: int
    ) -> QueryResult:
        """Execute a search query across multiple data sources."""
        
        # Build SQL conditions from parsed query
        conditions = []
        params = []
        
        # Add temporal filters
        temporal = parsed_query.get('temporal')
        if temporal:
            start_date, end_date = self._resolve_temporal_range(temporal)
            if start_date:
                conditions.append("timestamp >= ?")
                params.append(start_date.isoformat())
            if end_date:
                conditions.append("timestamp <= ?")
                params.append(end_date.isoformat())
        
        # Add entity filters
        entities = parsed_query.get('entities', [])
        for entity in entities:
            if entity['type'] == 'application':
                conditions.append("json_extract(window_info, '$.title') LIKE ?")
                params.append(f"%{entity['value']}%")
            elif entity['type'] == 'quoted_text':
                conditions.append("ocr_text LIKE ?")
                params.append(f"%{entity['value']}%")
            elif entity['type'] == 'file_extension':
                conditions.append("ocr_text LIKE ?")
                params.append(f"%.{entity['value']}%")
        
        # Add explicit filters
        filters = parsed_query.get('filters', [])
        for filter_cond in filters:
            if filter_cond.operator == 'contains':
                conditions.append(f"{filter_cond.field} LIKE ?")
                params.append(f"%{filter_cond.value}%")
            elif filter_cond.operator == 'eq':
                conditions.append(f"{filter_cond.field} = ?")
                params.append(filter_cond.value)
        
        # Build SQL query
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        sql = f"""
        SELECT * FROM screenshots 
        WHERE {where_clause}
        ORDER BY timestamp DESC 
        LIMIT ? OFFSET ?
        """
        
        params.extend([limit, offset])
        
        # Execute query
        screenshots = self.metadata_db.execute_query(sql, params)
        
        # Get total count
        count_sql = f"SELECT COUNT(*) FROM screenshots WHERE {where_clause}"
        total_count = self.metadata_db.execute_query(count_sql, params[:-2])[0][0]
        
        # Convert to dictionaries
        data = [dict(row) for row in screenshots]
        
        # Add semantic search if there's text in the query
        original_query = parsed_query.get('original_query', '')
        if original_query and hasattr(self.vector_db, 'search'):
            try:
                semantic_results = await asyncio.to_thread(
                    self.vector_db.search,
                    original_query,
                    limit=min(limit, 20)
                )
                
                # Merge results (prioritize SQL results, add semantic as additional)
                semantic_ids = {r.get('id') for r in semantic_results if 'id' in r}
                sql_ids = {r.get('id') for r in data if 'id' in r}
                
                for semantic_result in semantic_results:
                    if semantic_result.get('id') not in sql_ids:
                        data.append(semantic_result)
                        
            except Exception as e:
                self.logger.warning(f"Semantic search failed: {e}")
        
        return QueryResult(
            data=data,
            metadata={
                'query_type': 'search',
                'conditions_applied': len(conditions),
                'semantic_search_used': 'semantic_results' in locals()
            },
            execution_time_ms=0.0,  # Will be set by caller
            total_count=total_count
        )
    
    async def _execute_timeline_query(
        self,
        parsed_query: Dict[str, Any],
        limit: int,
        offset: int
    ) -> QueryResult:
        """Execute a timeline reconstruction query."""
        
        # Extract temporal range
        temporal = parsed_query.get('temporal')
        if temporal:
            start_date, end_date = self._resolve_temporal_range(temporal)
        else:
            # Default to last week
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
        
        # Get project timelines from analytics engine
        timelines = await asyncio.to_thread(
            self.analytics_engine.analyze_project_timelines,
            start_date,
            end_date
        )
        
        # Convert to timeline events
        timeline_data = []
        for timeline in timelines:
            for event in timeline.events:
                timeline_data.append({
                    'id': f"timeline_{timeline.project_id}_{event.timestamp.isoformat()}",
                    'timestamp': event.timestamp.isoformat(),
                    'event_type': event.event_type,
                    'application': event.application,
                    'title': event.title,
                    'description': event.description,
                    'project_id': event.project_id,
                    'project_name': timeline.name,
                    'confidence': event.confidence,
                    'metadata': event.metadata
                })
        
        # Sort by timestamp
        timeline_data.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Apply pagination
        paginated_data = timeline_data[offset:offset + limit]
        
        return QueryResult(
            data=paginated_data,
            metadata={
                'query_type': 'timeline',
                'projects_found': len(timelines),
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                }
            },
            execution_time_ms=0.0,
            total_count=len(timeline_data)
        )
    
    async def _execute_analytics_query(
        self,
        parsed_query: Dict[str, Any],
        limit: int,
        offset: int
    ) -> QueryResult:
        """Execute an analytics query."""
        
        # Extract temporal range
        temporal = parsed_query.get('temporal')
        if temporal:
            start_date, end_date = self._resolve_temporal_range(temporal)
        else:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Default to last month
        
        # Get analytics summary
        analytics_summary = await asyncio.to_thread(
            self.analytics_engine.get_analytics_summary,
            start_date,
            end_date
        )
        
        # Check for specific aggregations
        aggregation = parsed_query.get('aggregations')
        data = []
        
        if aggregation:
            # Process aggregation query
            if 'count' in aggregation.functions:
                if 'application' in aggregation.group_by:
                    # Count by application
                    app_counts = {}
                    for metric in analytics_summary['daily_productivity']:
                        for app in metric['applications_used']:
                            app_counts[app] = app_counts.get(app, 0) + 1
                    
                    data = [
                        {'application': app, 'count': count}
                        for app, count in sorted(app_counts.items(), key=lambda x: x[1], reverse=True)
                    ]
        else:
            # Return summary data
            data = [analytics_summary]
        
        return QueryResult(
            data=data[offset:offset + limit],
            metadata={
                'query_type': 'analytics',
                'analytics_period': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat(),
                    'days': (end_date - start_date).days
                }
            },
            execution_time_ms=0.0,
            total_count=len(data),
            aggregations=analytics_summary.get('summary', {})
        )
    
    async def _find_related_content(self, sample_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find content related to the sample results."""
        if not sample_data:
            return []
        
        related_results = []
        
        try:
            # Use vector search to find similar content
            for item in sample_data[:3]:  # Limit to first 3 items
                text_content = item.get('ocr_text', '')
                if text_content and len(text_content) > 10:
                    similar_items = await asyncio.to_thread(
                        self.vector_db.search,
                        text_content[:200],  # Use first 200 chars
                        limit=3
                    )
                    
                    for similar_item in similar_items:
                        if similar_item.get('id') != item.get('id'):
                            related_results.append(similar_item)
                            
        except Exception as e:
            self.logger.warning(f"Failed to find related content: {e}")
        
        # Remove duplicates and limit results
        seen_ids = set()
        unique_related = []
        for item in related_results:
            item_id = item.get('id')
            if item_id and item_id not in seen_ids:
                seen_ids.add(item_id)
                unique_related.append(item)
                if len(unique_related) >= 10:
                    break
        
        return unique_related
    
    def _resolve_temporal_range(self, temporal: TemporalQuery) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Resolve temporal query to specific datetime range."""
        if temporal.start_date and temporal.end_date:
            return temporal.start_date, temporal.end_date
        
        now = datetime.now()
        
        if temporal.time_window:
            if temporal.time_window == 'today':
                start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                end = now
            elif temporal.time_window == 'yesterday':
                start = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
                end = start + timedelta(days=1)
            elif temporal.time_window == 'last_week':
                start = now - timedelta(days=7)
                end = now
            elif temporal.time_window == 'last_month':
                start = now - timedelta(days=30)
                end = now
            elif temporal.time_window == 'this_week':
                days_since_monday = now.weekday()
                start = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_since_monday)
                end = now
            elif temporal.time_window == 'this_month':
                start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                end = now
            else:
                return None, None
            
            return start, end
        
        return temporal.start_date, temporal.end_date
    
    def _generate_cache_key(self, parsed_query: Dict[str, Any], limit: int, offset: int) -> str:
        """Generate a cache key for the query."""
        query_str = json.dumps(parsed_query, sort_keys=True, default=str)
        cache_data = f"{query_str}_{limit}_{offset}"
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear the query cache."""
        self.query_cache.clear()
        self.logger.info("Query cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = datetime.now()
        valid_entries = sum(
            1 for _, cache_time in self.query_cache.values()
            if now - cache_time < self.cache_ttl
        )
        
        return {
            'total_entries': len(self.query_cache),
            'valid_entries': valid_entries,
            'cache_hit_rate': valid_entries / max(1, len(self.query_cache))
        }


class QueryBuilder:
    """Helper class for building structured queries programmatically."""
    
    def __init__(self):
        self.conditions = []
        self.temporal = None
        self.aggregation = None
        self.intent = 'search'
    
    def add_condition(self, field: str, operator: str, value: Any, table: str = 'screenshots') -> 'QueryBuilder':
        """Add a condition to the query."""
        self.conditions.append(QueryCondition(field, operator, value, table))
        return self
    
    def set_temporal(self, start_date: datetime = None, end_date: datetime = None, time_window: str = None) -> 'QueryBuilder':
        """Set temporal constraints."""
        self.temporal = TemporalQuery(start_date, end_date, time_window)
        return self
    
    def set_aggregation(self, group_by: List[str], functions: List[str]) -> 'QueryBuilder':
        """Set aggregation parameters."""
        self.aggregation = AggregationQuery(group_by, functions)
        return self
    
    def set_intent(self, intent: str) -> 'QueryBuilder':
        """Set query intent."""
        self.intent = intent
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build the structured query."""
        query = {
            'intent': self.intent,
            'filters': self.conditions,
            'entities': [],
            'original_query': f"Programmatic query with {len(self.conditions)} conditions"
        }
        
        if self.temporal:
            query['temporal'] = self.temporal
        
        if self.aggregation:
            query['aggregations'] = self.aggregation
        
        return query