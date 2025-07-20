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
            
            # Run schema migrations for enhanced features
            self._run_schema_migrations(conn)
    
    def _run_schema_migrations(self, conn):
        """Run database schema migrations for enhanced features."""
        cursor = conn.cursor()
        
        # Check current schema version
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                description TEXT
            )
        """)
        
        # Get current version
        cursor.execute("SELECT MAX(version) FROM schema_version")
        result = cursor.fetchone()
        current_version = result[0] if result[0] is not None else 0
        
        migrations = [
            (1, "Add enhanced OCR fields", self._migrate_v1_enhanced_ocr),
            (2, "Add enhanced content analysis fields", self._migrate_v2_enhanced_analysis),
            (3, "Add domain-specific content tables", self._migrate_v3_domain_content),
        ]
        
        for version, description, migration_func in migrations:
            if version > current_version:
                self.logger.info(f"Running migration v{version}: {description}")
                try:
                    migration_func(cursor)
                    cursor.execute(
                        "INSERT INTO schema_version (version, description) VALUES (?, ?)",
                        (version, description)
                    )
                    conn.commit()
                    self.logger.info(f"Migration v{version} completed successfully")
                except Exception as e:
                    self.logger.error(f"Migration v{version} failed: {e}")
                    conn.rollback()
                    raise
    
    def _migrate_v1_enhanced_ocr(self, cursor):
        """Migration v1: Add enhanced OCR fields."""
        # Add new fields to ocr_results table
        new_fields = [
            ("urls", "TEXT"),  # JSON array of extracted URLs
            ("titles", "TEXT"),  # JSON array of extracted titles
            ("structured_content", "TEXT"),  # JSON object with content patterns
        ]
        
        # Check if fields already exist
        cursor.execute("PRAGMA table_info(ocr_results)")
        existing_fields = {row[1] for row in cursor.fetchall()}
        
        for field_name, field_type in new_fields:
            if field_name not in existing_fields:
                cursor.execute(f"ALTER TABLE ocr_results ADD COLUMN {field_name} {field_type}")
                self.logger.debug(f"Added field {field_name} to ocr_results table")
    
    def _migrate_v2_enhanced_analysis(self, cursor):
        """Migration v2: Add enhanced content analysis fields."""
        # Add new fields to content_analysis table
        new_fields = [
            ("domain_info", "TEXT"),  # JSON object with domain detection info
            ("video_info", "TEXT"),   # JSON object with video-specific info
            ("page_info", "TEXT"),    # JSON object with webpage info
            ("enhanced_metadata", "TEXT"),  # JSON object with enhanced metadata
        ]
        
        # Check if fields already exist
        cursor.execute("PRAGMA table_info(content_analysis)")
        existing_fields = {row[1] for row in cursor.fetchall()}
        
        for field_name, field_type in new_fields:
            if field_name not in existing_fields:
                cursor.execute(f"ALTER TABLE content_analysis ADD COLUMN {field_name} {field_type}")
                self.logger.debug(f"Added field {field_name} to content_analysis table")
    
    def _migrate_v3_domain_content(self, cursor):
        """Migration v3: Add domain-specific content tables and enhanced activity timeline."""
        # Create video content table for detailed video information
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS video_content (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                screenshot_id INTEGER NOT NULL,
                platform TEXT NOT NULL,  -- youtube, netflix, etc.
                video_title TEXT,
                channel_name TEXT,
                duration TEXT,
                view_count TEXT,
                upload_date TEXT,
                current_time TEXT,
                season TEXT,
                episode TEXT,
                genre TEXT,
                rating TEXT,
                description_snippet TEXT,
                metadata TEXT,  -- JSON for additional platform-specific data
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (screenshot_id) REFERENCES screenshots (id)
            )
        """)
        
        # Create webpage content table for detailed webpage information
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS webpage_content (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                screenshot_id INTEGER NOT NULL,
                url TEXT,
                title TEXT,
                domain TEXT,
                search_query TEXT,
                page_type TEXT,  -- browser, search_engine, social_media, etc.
                metadata TEXT,  -- JSON for additional webpage data
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (screenshot_id) REFERENCES screenshots (id)
            )
        """)
        
        # Create indexes for the new tables
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_video_screenshot_id ON video_content (screenshot_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_video_platform ON video_content (platform)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_video_title ON video_content (video_title)")
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_webpage_screenshot_id ON webpage_content (screenshot_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_webpage_domain ON webpage_content (domain)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_webpage_url ON webpage_content (url)")
        
        # Create full-text search for video titles and webpage content
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS video_search USING fts5(
                video_title,
                channel_name,
                description_snippet,
                content='video_content',
                content_rowid='id'
            )
        """)
        
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS webpage_search USING fts5(
                title,
                url,
                domain,
                search_query,
                content='webpage_content', 
                content_rowid='id'
            )
        """)
        
        # Create triggers for FTS maintenance
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS video_content_ai AFTER INSERT ON video_content BEGIN
                INSERT INTO video_search(rowid, video_title, channel_name, description_snippet) 
                VALUES (NEW.id, NEW.video_title, NEW.channel_name, NEW.description_snippet);
            END
        """)
        
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS webpage_content_ai AFTER INSERT ON webpage_content BEGIN
                INSERT INTO webpage_search(rowid, title, url, domain, search_query) 
                VALUES (NEW.id, NEW.title, NEW.url, NEW.domain, NEW.search_query);
            END
        """)
        
        # Enhanced activity timeline table for temporal queries
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS activity_timeline (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                screenshot_id INTEGER NOT NULL,
                activity_type TEXT NOT NULL,  -- video_view, webpage_visit, document_open, etc.
                domain TEXT,  -- youtube.com, netflix.com, etc.
                platform TEXT,  -- youtube, netflix, browser, etc.
                title TEXT,  -- video title, page title, etc.
                url TEXT,  -- full URL if applicable
                duration_seconds INTEGER,  -- how long activity was active
                session_start DATETIME NOT NULL,
                session_end DATETIME,
                metadata TEXT,  -- JSON with activity-specific data
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (screenshot_id) REFERENCES screenshots (id)
            )
        """)
        
        # Create indexes for temporal queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timeline_session_start ON activity_timeline (session_start)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timeline_domain ON activity_timeline (domain)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timeline_platform ON activity_timeline (platform)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timeline_activity_type ON activity_timeline (activity_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timeline_title ON activity_timeline (title)")
        
        # Create compound indexes for common temporal queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timeline_platform_time ON activity_timeline (platform, session_start)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timeline_domain_time ON activity_timeline (domain, session_start)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timeline_type_time ON activity_timeline (activity_type, session_start)")
        
        # FTS for activity timeline content
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS activity_search USING fts5(
                title,
                url,
                metadata,
                content='activity_timeline',
                content_rowid='id'
            )
        """)
        
        # Trigger for FTS maintenance
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS activity_timeline_ai AFTER INSERT ON activity_timeline BEGIN
                INSERT INTO activity_search(rowid, title, url, metadata) 
                VALUES (NEW.id, NEW.title, NEW.url, NEW.metadata);
            END
        """)
    
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
                'regions': getattr(ocr_result, 'regions', []),
                'urls': getattr(ocr_result, 'urls', []),
                'titles': getattr(ocr_result, 'titles', []),
                'structured_content': getattr(ocr_result, 'structured_content', {})
            }
        else:
            # Dictionary
            ocr_data = ocr_result
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO ocr_results 
                (screenshot_id, text, confidence, language, method, word_count, regions, urls, titles, structured_content)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                screenshot_id,
                ocr_data['text'],
                ocr_data['confidence'],
                ocr_data.get('language'),
                ocr_data.get('method'),
                ocr_data.get('word_count', 0),
                json.dumps(ocr_data.get('regions', [])),
                json.dumps(ocr_data.get('urls', [])),
                json.dumps(ocr_data.get('titles', [])),
                json.dumps(ocr_data.get('structured_content', {}))
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
                'metadata': getattr(analysis_result, 'metadata', {}),
                'domain_info': getattr(analysis_result, 'domain_info', {}),
                'video_info': getattr(analysis_result, 'video_info', {}),
                'page_info': getattr(analysis_result, 'page_info', {}),
                'enhanced_metadata': getattr(analysis_result, 'enhanced_metadata', {})
            }
        else:
            # Dictionary
            analysis_data = analysis_result
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO content_analysis 
                (screenshot_id, content_type, description, confidence, tags, ui_elements, metadata, 
                 domain_info, video_info, page_info, enhanced_metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                screenshot_id,
                analysis_data['content_type'],
                analysis_data.get('description', ''),
                analysis_data['confidence'],
                json.dumps(analysis_data.get('tags', [])),
                json.dumps(analysis_data.get('ui_elements', [])),
                json.dumps(analysis_data.get('metadata', {})),
                json.dumps(analysis_data.get('domain_info', {})),
                json.dumps(analysis_data.get('video_info', {})),
                json.dumps(analysis_data.get('page_info', {})),
                json.dumps(analysis_data.get('enhanced_metadata', {}))
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
    
    @log_performance
    def store_video_content(self, screenshot_id: int, video_data: Dict[str, Any]) -> int:
        """
        Store video content information.
        
        Args:
            screenshot_id: ID of the associated screenshot.
            video_data: Dictionary containing video information.
            
        Returns:
            int: Video content ID.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO video_content 
                (screenshot_id, platform, video_title, channel_name, duration, view_count, 
                 upload_date, current_time, season, episode, genre, rating, description_snippet, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                screenshot_id,
                video_data.get('platform', 'unknown'),
                video_data.get('video_title') or video_data.get('title', ''),
                video_data.get('channel_name', ''),
                video_data.get('duration', ''),
                video_data.get('view_count', ''),
                video_data.get('upload_date', ''),
                video_data.get('current_time', ''),
                video_data.get('season', ''),
                video_data.get('episode', ''),
                video_data.get('genre', ''),
                video_data.get('rating', ''),
                video_data.get('description_snippet', ''),
                json.dumps({k: v for k, v in video_data.items() if k not in [
                    'platform', 'video_title', 'title', 'channel_name', 'duration', 
                    'view_count', 'upload_date', 'current_time', 'season', 'episode', 
                    'genre', 'rating', 'description_snippet'
                ]})
            ))
            
            video_id = cursor.lastrowid
            conn.commit()
            
            return video_id
    
    @log_performance
    def store_webpage_content(self, screenshot_id: int, webpage_data: Dict[str, Any]) -> int:
        """
        Store webpage content information.
        
        Args:
            screenshot_id: ID of the associated screenshot.
            webpage_data: Dictionary containing webpage information.
            
        Returns:
            int: Webpage content ID.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO webpage_content 
                (screenshot_id, url, title, domain, search_query, page_type, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                screenshot_id,
                webpage_data.get('url', ''),
                webpage_data.get('title', ''),
                webpage_data.get('domain', ''),
                webpage_data.get('search_query', ''),
                webpage_data.get('type') or webpage_data.get('page_type', 'webpage'),
                json.dumps({k: v for k, v in webpage_data.items() if k not in [
                    'url', 'title', 'domain', 'search_query', 'type', 'page_type'
                ]})
            ))
            
            webpage_id = cursor.lastrowid
            conn.commit()
            
            return webpage_id
    
    def get_video_content_by_screenshot(self, screenshot_id: int) -> Optional[Dict[str, Any]]:
        """Get video content for a screenshot."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM video_content WHERE screenshot_id = ?", (screenshot_id,))
            row = cursor.fetchone()
            
            if row:
                result = dict(row)
                # Parse JSON metadata
                if result.get('metadata'):
                    try:
                        result['metadata'] = json.loads(result['metadata'])
                    except json.JSONDecodeError:
                        result['metadata'] = {}
                return result
            return None
    
    def get_webpage_content_by_screenshot(self, screenshot_id: int) -> Optional[Dict[str, Any]]:
        """Get webpage content for a screenshot."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM webpage_content WHERE screenshot_id = ?", (screenshot_id,))
            row = cursor.fetchone()
            
            if row:
                result = dict(row)
                # Parse JSON metadata
                if result.get('metadata'):
                    try:
                        result['metadata'] = json.loads(result['metadata'])
                    except json.JSONDecodeError:
                        result['metadata'] = {}
                return result
            return None
    
    def search_video_content(self, query: str, platform: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Search video content using FTS."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if platform:
                cursor.execute("""
                    SELECT v.*, s.timestamp, s.file_path 
                    FROM video_search vs
                    JOIN video_content v ON vs.rowid = v.id
                    JOIN screenshots s ON v.screenshot_id = s.id
                    WHERE video_search MATCH ? AND v.platform = ?
                    ORDER BY rank LIMIT ?
                """, (query, platform, limit))
            else:
                cursor.execute("""
                    SELECT v.*, s.timestamp, s.file_path 
                    FROM video_search vs
                    JOIN video_content v ON vs.rowid = v.id
                    JOIN screenshots s ON v.screenshot_id = s.id
                    WHERE video_search MATCH ?
                    ORDER BY rank LIMIT ?
                """, (query, limit))
            
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                if result.get('metadata'):
                    try:
                        result['metadata'] = json.loads(result['metadata'])
                    except json.JSONDecodeError:
                        result['metadata'] = {}
                results.append(result)
            
            return results
    
    def search_webpage_content(self, query: str, domain: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Search webpage content using FTS."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if domain:
                cursor.execute("""
                    SELECT w.*, s.timestamp, s.file_path 
                    FROM webpage_search ws
                    JOIN webpage_content w ON ws.rowid = w.id
                    JOIN screenshots s ON w.screenshot_id = s.id
                    WHERE webpage_search MATCH ? AND w.domain LIKE ?
                    ORDER BY rank LIMIT ?
                """, (query, f"%{domain}%", limit))
            else:
                cursor.execute("""
                    SELECT w.*, s.timestamp, s.file_path 
                    FROM webpage_search ws
                    JOIN webpage_content w ON ws.rowid = w.id
                    JOIN screenshots s ON w.screenshot_id = s.id
                    WHERE webpage_search MATCH ?
                    ORDER BY rank LIMIT ?
                """, (query, limit))
            
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                if result.get('metadata'):
                    try:
                        result['metadata'] = json.loads(result['metadata'])
                    except json.JSONDecodeError:
                        result['metadata'] = {}
                results.append(result)
            
            return results
    
    @log_performance
    def log_activity_timeline(self, screenshot_id: int, activity_data: Dict[str, Any]) -> int:
        """
        Log an activity to the timeline for temporal queries.
        
        Args:
            screenshot_id: ID of the associated screenshot.
            activity_data: Dictionary containing activity information.
            
        Returns:
            int: Activity timeline ID.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Determine session_start from timestamp or use current time
            session_start = activity_data.get('session_start') or activity_data.get('timestamp') or datetime.now().isoformat()
            if isinstance(session_start, datetime):
                session_start = session_start.isoformat()
            
            cursor.execute("""
                INSERT INTO activity_timeline 
                (screenshot_id, activity_type, domain, platform, title, url, 
                 duration_seconds, session_start, session_end, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                screenshot_id,
                activity_data.get('activity_type', 'unknown'),
                activity_data.get('domain', ''),
                activity_data.get('platform', ''),
                activity_data.get('title', ''),
                activity_data.get('url', ''),
                activity_data.get('duration_seconds'),
                session_start,
                activity_data.get('session_end'),
                json.dumps({k: v for k, v in activity_data.items() if k not in [
                    'activity_type', 'domain', 'platform', 'title', 'url', 
                    'duration_seconds', 'session_start', 'session_end', 'timestamp'
                ]})
            ))
            
            activity_id = cursor.lastrowid
            conn.commit()
            
            return activity_id
    
    def get_last_activity_by_platform(self, platform: str, limit: int = 1) -> List[Dict[str, Any]]:
        """
        Get the most recent activity for a specific platform.
        
        Args:
            platform: Platform name (youtube, netflix, etc.)
            limit: Number of results to return
            
        Returns:
            List of activity dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT a.*, s.timestamp as screenshot_timestamp, s.file_path
                FROM activity_timeline a
                JOIN screenshots s ON a.screenshot_id = s.id
                WHERE a.platform = ?
                ORDER BY a.session_start DESC
                LIMIT ?
            """, (platform, limit))
            
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                if result.get('metadata'):
                    try:
                        result['metadata'] = json.loads(result['metadata'])
                    except json.JSONDecodeError:
                        result['metadata'] = {}
                results.append(result)
            
            return results
    
    def get_last_activity_by_domain(self, domain: str, limit: int = 1) -> List[Dict[str, Any]]:
        """Get the most recent activity for a specific domain."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT a.*, s.timestamp as screenshot_timestamp, s.file_path
                FROM activity_timeline a
                JOIN screenshots s ON a.screenshot_id = s.id
                WHERE a.domain LIKE ?
                ORDER BY a.session_start DESC
                LIMIT ?
            """, (f"%{domain}%", limit))
            
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                if result.get('metadata'):
                    try:
                        result['metadata'] = json.loads(result['metadata'])
                    except json.JSONDecodeError:
                        result['metadata'] = {}
                results.append(result)
            
            return results
    
    def get_activities_by_time_range(self, start_time: datetime, end_time: datetime, 
                                   platform: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get activities within a specific time range."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            start_str = start_time.isoformat() if isinstance(start_time, datetime) else start_time
            end_str = end_time.isoformat() if isinstance(end_time, datetime) else end_time
            
            if platform:
                cursor.execute("""
                    SELECT a.*, s.timestamp as screenshot_timestamp, s.file_path
                    FROM activity_timeline a
                    JOIN screenshots s ON a.screenshot_id = s.id
                    WHERE a.session_start BETWEEN ? AND ? AND a.platform = ?
                    ORDER BY a.session_start DESC
                    LIMIT ?
                """, (start_str, end_str, platform, limit))
            else:
                cursor.execute("""
                    SELECT a.*, s.timestamp as screenshot_timestamp, s.file_path
                    FROM activity_timeline a
                    JOIN screenshots s ON a.screenshot_id = s.id
                    WHERE a.session_start BETWEEN ? AND ?
                    ORDER BY a.session_start DESC
                    LIMIT ?
                """, (start_str, end_str, limit))
            
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                if result.get('metadata'):
                    try:
                        result['metadata'] = json.loads(result['metadata'])
                    except json.JSONDecodeError:
                        result['metadata'] = {}
                results.append(result)
            
            return results
    
    def search_activity_timeline(self, query: str, platform: Optional[str] = None, 
                                limit: int = 10) -> List[Dict[str, Any]]:
        """Search activity timeline using FTS."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if platform:
                cursor.execute("""
                    SELECT a.*, s.timestamp as screenshot_timestamp, s.file_path
                    FROM activity_search ats
                    JOIN activity_timeline a ON ats.rowid = a.id
                    JOIN screenshots s ON a.screenshot_id = s.id
                    WHERE activity_search MATCH ? AND a.platform = ?
                    ORDER BY a.session_start DESC
                    LIMIT ?
                """, (query, platform, limit))
            else:
                cursor.execute("""
                    SELECT a.*, s.timestamp as screenshot_timestamp, s.file_path
                    FROM activity_search ats
                    JOIN activity_timeline a ON ats.rowid = a.id
                    JOIN screenshots s ON a.screenshot_id = s.id
                    WHERE activity_search MATCH ?
                    ORDER BY a.session_start DESC
                    LIMIT ?
                """, (query, limit))
            
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                if result.get('metadata'):
                    try:
                        result['metadata'] = json.loads(result['metadata'])
                    except json.JSONDecodeError:
                        result['metadata'] = {}
                results.append(result)
            
            return results
    
    def get_recent_video_activities(self, platform: Optional[str] = None, hours: int = 24, 
                                  limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent video viewing activities."""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if platform:
                cursor.execute("""
                    SELECT a.*, s.timestamp as screenshot_timestamp, s.file_path,
                           v.video_title, v.channel_name, v.duration, v.view_count
                    FROM activity_timeline a
                    JOIN screenshots s ON a.screenshot_id = s.id
                    LEFT JOIN video_content v ON a.screenshot_id = v.screenshot_id
                    WHERE a.activity_type = 'video_view' 
                      AND a.session_start BETWEEN ? AND ?
                      AND a.platform = ?
                    ORDER BY a.session_start DESC
                    LIMIT ?
                """, (start_time.isoformat(), end_time.isoformat(), platform, limit))
            else:
                cursor.execute("""
                    SELECT a.*, s.timestamp as screenshot_timestamp, s.file_path,
                           v.video_title, v.channel_name, v.duration, v.view_count
                    FROM activity_timeline a
                    JOIN screenshots s ON a.screenshot_id = s.id
                    LEFT JOIN video_content v ON a.screenshot_id = v.screenshot_id
                    WHERE a.activity_type = 'video_view' 
                      AND a.session_start BETWEEN ? AND ?
                    ORDER BY a.session_start DESC
                    LIMIT ?
                """, (start_time.isoformat(), end_time.isoformat(), limit))
            
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                if result.get('metadata'):
                    try:
                        result['metadata'] = json.loads(result['metadata'])
                    except json.JSONDecodeError:
                        result['metadata'] = {}
                results.append(result)
            
            return results
    
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