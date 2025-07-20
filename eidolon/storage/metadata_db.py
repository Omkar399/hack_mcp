"""
Metadata Database for structured data storage

SQLite database for storing screenshot metadata, OCR results, and content analysis.
"""

import sqlite3
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from contextlib import contextmanager

from ..utils.logging import get_component_logger, log_performance, log_exceptions
from ..utils.config import get_config


class MetadataDatabase:
    """SQLite database for storing screenshot metadata and analysis results."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.config = get_config()
        self.logger = get_component_logger("storage.metadata_db")
        
        # Set database path
        if db_path:
            self.db_path = Path(db_path)
        else:
            self.db_path = Path(self.config.memory.db_path)
        
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        self.logger.info(f"Metadata database initialized: {self.db_path}")
    
    @contextmanager
    def get_connection(self):
        """Get database connection with automatic cleanup."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
        finally:
            conn.close()
    
    def _init_database(self):
        """Initialize database schema."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Screenshots table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS screenshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hash TEXT UNIQUE NOT NULL,
                    file_path TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    size_bytes INTEGER,
                    window_info TEXT,  -- JSON
                    monitor_info TEXT,  -- JSON
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # OCR results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ocr_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    screenshot_id INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    language TEXT,
                    method TEXT,
                    word_count INTEGER,
                    regions TEXT,  -- JSON array of text regions
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (screenshot_id) REFERENCES screenshots (id)
                )
            """)
            
            # Content analysis table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS content_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    screenshot_id INTEGER NOT NULL,
                    content_type TEXT NOT NULL,
                    description TEXT,
                    confidence REAL NOT NULL,
                    tags TEXT,  -- JSON array
                    ui_elements TEXT,  -- JSON array
                    metadata TEXT,  -- JSON object
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (screenshot_id) REFERENCES screenshots (id)
                )
            """)
            
            # Activity events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS activity_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,  -- screenshot, ocr, analysis, etc.
                    screenshot_id INTEGER,
                    event_data TEXT,  -- JSON
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (screenshot_id) REFERENCES screenshots (id)
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_screenshots_timestamp ON screenshots (timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_screenshots_hash ON screenshots (hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ocr_screenshot_id ON ocr_results (screenshot_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_analysis_screenshot_id ON content_analysis (screenshot_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_activity_timestamp ON activity_events (timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_activity_type ON activity_events (event_type)")
            
            # Full-text search for OCR text
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS ocr_search USING fts5(
                    text,
                    content='ocr_results',
                    content_rowid='id'
                )
            """)
            
            # Create triggers to maintain FTS
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS ocr_results_ai AFTER INSERT ON ocr_results BEGIN
                    INSERT INTO ocr_search(rowid, text) VALUES (NEW.id, NEW.text);
                END
            """)
            
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS ocr_results_ad AFTER DELETE ON ocr_results BEGIN
                    INSERT INTO ocr_search(ocr_search, rowid, text) VALUES('delete', OLD.id, OLD.text);
                END
            """)
            
            cursor.execute("""
                CREATE TRIGGER IF NOT EXISTS ocr_results_au AFTER UPDATE ON ocr_results BEGIN
                    INSERT INTO ocr_search(ocr_search, rowid, text) VALUES('delete', OLD.id, OLD.text);
                    INSERT INTO ocr_search(rowid, text) VALUES (NEW.id, NEW.text);
                END
            """)
            
            conn.commit()
    
    @log_performance
    def store_screenshot(self, file_path: str = None, timestamp: datetime = None, hash: str = None, 
                       size: Tuple[int, int] = None, file_size: int = None, 
                       screenshot_data: Dict[str, Any] = None, **kwargs) -> int:
        """
        Store screenshot metadata.
        
        Args:
            file_path: Path to screenshot file
            timestamp: Screenshot timestamp
            hash: Screenshot hash
            size: Screenshot dimensions (width, height)
            file_size: File size in bytes
            screenshot_data: Screenshot metadata dictionary (legacy support)
            **kwargs: Additional metadata
            
        Returns:
            int: Screenshot ID.
        """
        # Handle both new signature and legacy dictionary approach
        if screenshot_data is not None:
            # Legacy dictionary approach
            data = screenshot_data
        else:
            # New parameter approach
            data = {
                'file_path': file_path,
                'timestamp': timestamp,
                'hash': hash,
                'size': size,
                'file_size': file_size or kwargs.get('size_bytes'),
                **kwargs
            }
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Ensure timestamp is a string for SQLite
            timestamp_str = data.get('timestamp')
            if timestamp_str is None:
                timestamp_str = datetime.now().isoformat()
            elif isinstance(timestamp_str, datetime):
                timestamp_str = timestamp_str.isoformat()
            elif not isinstance(timestamp_str, str):
                timestamp_str = str(timestamp_str)
            
            # Handle size tuple/list - ensure we have valid dimensions
            width = None
            height = None
            if data.get('size'):
                if isinstance(data['size'], (list, tuple)) and len(data['size']) >= 2:
                    width = int(data['size'][0]) if data['size'][0] else 0
                    height = int(data['size'][1]) if data['size'][1] else 0
            
            # Fallback for missing dimensions
            if width is None or height is None:
                width = width or 1920  # Default width
                height = height or 1080  # Default height
            
            # Validate required fields
            hash_val = str(data.get('hash', ''))
            if not hash_val:
                hash_val = f"missing_hash_{int(time.time())}"
            
            file_path_val = str(data.get('file_path', ''))
            
            cursor.execute("""
                INSERT OR REPLACE INTO screenshots 
                (hash, file_path, timestamp, width, height, size_bytes, window_info, monitor_info)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                hash_val,
                file_path_val,
                timestamp_str,
                width,
                height,
                data.get('file_size') or data.get('size_bytes'),
                json.dumps(data.get('window_info', {})),
                json.dumps(data.get('monitor_info', {}))
            ))
            
            screenshot_id = cursor.lastrowid
            conn.commit()
            
            # Log activity
            timestamp_str = data['timestamp'].isoformat() if isinstance(data['timestamp'], datetime) else str(data['timestamp'])
            self.log_activity('screenshot', screenshot_id, timestamp_str)
            
            return screenshot_id
    
    @log_performance
    def store_ocr_result(self, screenshot_id: int, ocr_result) -> int:
        """
        Store OCR result.
        
        Args:
            screenshot_id: ID of the associated screenshot.
            ocr_result: OCR result object or dictionary.
            
        Returns:
            int: OCR result ID.
        """
        # Handle both object and dictionary input
        if hasattr(ocr_result, '__dict__'):
            # Object with attributes
            ocr_data = {
                'text': getattr(ocr_result, 'text', ''),
                'confidence': getattr(ocr_result, 'confidence', 0.0),
                'language': getattr(ocr_result, 'language', None),
                'method': getattr(ocr_result, 'method', None),
                'word_count': getattr(ocr_result, 'word_count', 0),
                'regions': getattr(ocr_result, 'regions', [])
            }
        else:
            # Dictionary
            ocr_data = ocr_result
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO ocr_results 
                (screenshot_id, text, confidence, language, method, word_count, regions)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                screenshot_id,
                ocr_data['text'],
                ocr_data['confidence'],
                ocr_data.get('language'),
                ocr_data.get('method'),
                ocr_data.get('word_count', 0),
                json.dumps(ocr_data.get('regions', []))
            ))
            
            ocr_id = cursor.lastrowid
            conn.commit()
            
            # Log activity
            self.log_activity('ocr', screenshot_id, datetime.now().isoformat())
            
            return ocr_id
    
    @log_performance
    def store_content_analysis(self, screenshot_id: int, analysis_result) -> int:
        """
        Store content analysis result.
        
        Args:
            screenshot_id: ID of the associated screenshot.
            analysis_result: Content analysis object or dictionary.
            
        Returns:
            int: Content analysis ID.
        """
        # Handle both object and dictionary input
        if hasattr(analysis_result, '__dict__'):
            # Object with attributes
            analysis_data = {
                'content_type': getattr(analysis_result, 'content_type', 'unknown'),
                'description': getattr(analysis_result, 'description', ''),
                'confidence': getattr(analysis_result, 'confidence', 0.0),
                'tags': getattr(analysis_result, 'tags', []),
                'ui_elements': getattr(analysis_result, 'ui_elements', []),
                'metadata': getattr(analysis_result, 'metadata', {})
            }
        else:
            # Dictionary
            analysis_data = analysis_result
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO content_analysis 
                (screenshot_id, content_type, description, confidence, tags, ui_elements, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                screenshot_id,
                analysis_data['content_type'],
                analysis_data.get('description', ''),
                analysis_data['confidence'],
                json.dumps(analysis_data.get('tags', [])),
                json.dumps(analysis_data.get('ui_elements', [])),
                json.dumps(analysis_data.get('metadata', {}))
            ))
            
            analysis_id = cursor.lastrowid
            conn.commit()
            
            # Log activity
            self.log_activity('analysis', screenshot_id, datetime.now().isoformat())
            
            return analysis_id
    
    def log_activity(self, event_type: str, screenshot_id: Optional[int], timestamp: str, event_data: Optional[Dict[str, Any]] = None):
        """Log an activity event."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO activity_events (event_type, screenshot_id, event_data, timestamp)
                VALUES (?, ?, ?, ?)
            """, (
                event_type,
                screenshot_id,
                json.dumps(event_data) if event_data else None,
                timestamp
            ))
            
            conn.commit()
    
    @log_performance
    def search_text(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search through OCR text using full-text search.
        
        Args:
            query: Search query.
            limit: Maximum number of results.
            
        Returns:
            List of search results.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Use FTS for text search
            cursor.execute("""
                SELECT 
                    s.id as screenshot_id,
                    s.hash,
                    s.file_path,
                    s.timestamp,
                    o.text,
                    o.confidence,
                    o.method,
                    ca.content_type,
                    ca.description,
                    ca.tags
                FROM ocr_search 
                JOIN ocr_results o ON ocr_search.rowid = o.id
                JOIN screenshots s ON o.screenshot_id = s.id
                LEFT JOIN content_analysis ca ON s.id = ca.screenshot_id
                WHERE ocr_search MATCH ?
                ORDER BY s.timestamp DESC
                LIMIT ?
            """, (query, limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'screenshot_id': row['screenshot_id'],
                    'hash': row['hash'],
                    'file_path': row['file_path'],
                    'timestamp': row['timestamp'],
                    'text': row['text'],
                    'ocr_confidence': row['confidence'],
                    'ocr_method': row['method'],
                    'content_type': row['content_type'],
                    'description': row['description'],
                    'tags': json.loads(row['tags']) if row['tags'] else []
                })
            
            return results
    
    def search_by_content_type(self, content_type: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search screenshots by content type.
        
        Args:
            content_type: Content type to search for.
            limit: Maximum number of results.
            
        Returns:
            List of search results.
        """
        return self.get_content_by_type(content_type, limit)
    
    def search_by_time_range(self, start_time: datetime, end_time: datetime, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search screenshots by time range.
        
        Args:
            start_time: Start of time range.
            end_time: End of time range.
            limit: Maximum number of results.
            
        Returns:
            List of search results.
        """
        results = self.get_screenshots_by_timerange(start_time, end_time)
        return results[:limit]
    
    def search_combined(self, text_query: str = None, content_types: List[str] = None, 
                       start_time: datetime = None, end_time: datetime = None, 
                       limit: int = 50) -> List[Dict[str, Any]]:
        """
        Combined search with multiple criteria.
        
        Args:
            text_query: Text to search for.
            content_types: List of content types to filter by.
            start_time: Start of time range.
            end_time: End of time range.
            limit: Maximum number of results.
            
        Returns:
            List of search results.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Build dynamic query
            conditions = []
            params = []
            
            base_query = """
                SELECT DISTINCT
                    s.id as screenshot_id,
                    s.hash,
                    s.file_path,
                    s.timestamp,
                    o.text,
                    o.confidence as ocr_confidence,
                    o.method as ocr_method,
                    ca.content_type,
                    ca.description,
                    ca.tags
                FROM screenshots s
                LEFT JOIN ocr_results o ON s.id = o.screenshot_id
                LEFT JOIN content_analysis ca ON s.id = ca.screenshot_id
            """
            
            # Add text search condition
            if text_query:
                base_query = """
                    SELECT DISTINCT
                        s.id as screenshot_id,
                        s.hash,
                        s.file_path,
                        s.timestamp,
                        o.text,
                        o.confidence as ocr_confidence,
                        o.method as ocr_method,
                        ca.content_type,
                        ca.description,
                        ca.tags
                    FROM ocr_search 
                    JOIN ocr_results o ON ocr_search.rowid = o.id
                    JOIN screenshots s ON o.screenshot_id = s.id
                    LEFT JOIN content_analysis ca ON s.id = ca.screenshot_id
                """
                conditions.append("ocr_search MATCH ?")
                params.append(text_query)
            
            # Add content type filter
            if content_types:
                placeholders = ','.join(['?'] * len(content_types))
                conditions.append(f"ca.content_type IN ({placeholders})")
                params.extend(content_types)
            
            # Add time range filter
            if start_time:
                conditions.append("s.timestamp >= ?")
                params.append(start_time.isoformat())
            if end_time:
                conditions.append("s.timestamp <= ?")
                params.append(end_time.isoformat())
            
            # Combine conditions
            if conditions:
                base_query += " WHERE " + " AND ".join(conditions)
            
            base_query += " ORDER BY s.timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(base_query, params)
            
            results = []
            for row in cursor.fetchall():
                result = {
                    'screenshot_id': row['screenshot_id'],
                    'hash': row['hash'],
                    'file_path': row['file_path'],
                    'timestamp': row['timestamp'],
                    'text': row['text'],
                    'ocr_confidence': row['ocr_confidence'],
                    'ocr_method': row['ocr_method'],
                    'content_type': row['content_type'],
                    'description': row['description'],
                    'tags': json.loads(row['tags']) if row['tags'] else []
                }
                results.append(result)
            
            return results
    
    def get_screenshots_by_timerange(self, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Get screenshots within a time range."""
        return self.get_screenshots_by_time_range(start_time, end_time)
    
    def get_screenshots_by_time_range(self, start_time: datetime, end_time: datetime, limit: int = 50) -> List[Dict[str, Any]]:
        """Get screenshots within a time range (with underscores for compatibility)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT s.*, ca.content_type, ca.description
                FROM screenshots s
                LEFT JOIN content_analysis ca ON s.id = ca.screenshot_id
                WHERE s.timestamp BETWEEN ? AND ?
                ORDER BY s.timestamp DESC
                LIMIT ?
            """, (start_time.isoformat(), end_time.isoformat(), limit))
            
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                # Parse JSON fields
                if result.get('window_info'):
                    result['window_info'] = json.loads(result['window_info'])
                if result.get('monitor_info'):
                    result['monitor_info'] = json.loads(result['monitor_info'])
                results.append(result)
            
            return results
    
    def get_content_by_type(self, content_type: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get screenshots by content type."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    s.id,
                    s.hash,
                    s.file_path,
                    s.timestamp,
                    ca.content_type,
                    ca.description,
                    ca.confidence,
                    ca.tags
                FROM screenshots s
                JOIN content_analysis ca ON s.id = ca.screenshot_id
                WHERE ca.content_type = ?
                ORDER BY s.timestamp DESC
                LIMIT ?
            """, (content_type, limit))
            
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                result['tags'] = json.loads(result['tags']) if result['tags'] else []
                results.append(result)
            
            return results
    
    def get_content_analysis(self, screenshot_id: int) -> Optional[Dict[str, Any]]:
        """Get content analysis for a screenshot."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM content_analysis 
                WHERE screenshot_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (screenshot_id,))
            
            row = cursor.fetchone()
            if row:
                result = dict(row)
                # Parse JSON fields
                if result.get('tags'):
                    result['tags'] = json.loads(result['tags'])
                if result.get('ui_elements'):
                    result['ui_elements'] = json.loads(result['ui_elements'])
                if result.get('metadata'):
                    result['metadata'] = json.loads(result['metadata'])
                return result
            return None
    
    def get_ocr_result(self, screenshot_id: int) -> Optional[Dict[str, Any]]:
        """Get OCR result for a screenshot."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM ocr_results 
                WHERE screenshot_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (screenshot_id,))
            
            row = cursor.fetchone()
            if row:
                result = dict(row)
                # Parse JSON fields
                if result.get('regions'):
                    result['regions'] = json.loads(result['regions'])
                return result
            return None
    
    def cleanup_old_screenshots(self, days_to_keep: int) -> int:
        """Clean up old screenshots (alias for cleanup_old_data)."""
        return self.cleanup_old_data(days_to_keep)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Total counts
            cursor.execute("SELECT COUNT(*) FROM screenshots")
            total_screenshots = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM ocr_results WHERE text != ''")
            screenshots_with_text = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM content_analysis")
            analyzed_screenshots = cursor.fetchone()[0]
            
            # Date range
            cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM screenshots")
            date_range = cursor.fetchone()
            
            # Content type distribution
            cursor.execute("""
                SELECT content_type, COUNT(*) as count 
                FROM content_analysis 
                GROUP BY content_type 
                ORDER BY count DESC
            """)
            content_types = dict(cursor.fetchall())
            
            # Recent activity
            cursor.execute("""
                SELECT DATE(timestamp) as date, COUNT(*) as count
                FROM screenshots 
                WHERE timestamp >= datetime('now', '-7 days')
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            """)
            recent_activity = dict(cursor.fetchall())
            
            return {
                'total_screenshots': total_screenshots,
                'screenshots_with_text': screenshots_with_text,
                'analyzed_screenshots': analyzed_screenshots,
                'earliest_screenshot': date_range[0],
                'latest_screenshot': date_range[1],
                'content_types': content_types,
                'recent_activity': recent_activity
            }
    
    @log_performance
    def cleanup_old_data(self, days_to_keep: int) -> int:
        """
        Clean up data older than specified days.
        
        Args:
            days_to_keep: Number of days to keep.
            
        Returns:
            int: Number of records deleted.
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get screenshots to delete
            cursor.execute("""
                SELECT id FROM screenshots 
                WHERE timestamp < ?
            """, (cutoff_date.isoformat(),))
            
            screenshot_ids = [row[0] for row in cursor.fetchall()]
            
            if not screenshot_ids:
                return 0
            
            # Delete related records
            placeholders = ','.join(['?'] * len(screenshot_ids))
            
            cursor.execute(f"DELETE FROM activity_events WHERE screenshot_id IN ({placeholders})", screenshot_ids)
            cursor.execute(f"DELETE FROM content_analysis WHERE screenshot_id IN ({placeholders})", screenshot_ids)
            cursor.execute(f"DELETE FROM ocr_results WHERE screenshot_id IN ({placeholders})", screenshot_ids)
            cursor.execute(f"DELETE FROM screenshots WHERE id IN ({placeholders})", screenshot_ids)
            
            deleted_count = len(screenshot_ids)
            conn.commit()
            
            # Vacuum to reclaim space
            cursor.execute("VACUUM")
            
            self.logger.info(f"Cleaned up {deleted_count} old records")
            return deleted_count
    
    def execute_query(self, sql: str, params: Optional[List[Any]] = None) -> List[sqlite3.Row]:
        """
        Execute a custom SQL query and return results.
        
        Args:
            sql: SQL query string
            params: Query parameters
            
        Returns:
            List of query results as Row objects
        """
        if params is None:
            params = []
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            
            if sql.strip().upper().startswith('SELECT'):
                return cursor.fetchall()
            else:
                conn.commit()
                return []