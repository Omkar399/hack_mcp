#!/usr/bin/env python3
"""
Unit tests for the Metadata Database component.

Tests cover SQLite database operations, screenshot metadata storage,
OCR results management, and content analysis persistence.
"""

import pytest
import tempfile
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import Dict, Any

# Import the modules to test
from eidolon.storage.metadata_db import MetadataDatabase


class TestMetadataDatabaseInit:
    """Test MetadataDatabase initialization and setup."""
    
    def test_metadata_db_creation(self):
        """Test metadata database creation with custom path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_metadata.db"
            
            with patch('eidolon.storage.metadata_db.get_config'), \
                 patch('eidolon.storage.metadata_db.get_component_logger'):
                
                db = MetadataDatabase(str(db_path))
                
                # Check database file was created
                assert db_path.exists()
                assert db.db_path == db_path
    
    def test_database_schema_creation(self):
        """Test that all required tables are created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_schema.db"
            
            with patch('eidolon.storage.metadata_db.get_config'), \
                 patch('eidolon.storage.metadata_db.get_component_logger'):
                
                db = MetadataDatabase(str(db_path))
                
                # Check tables exist
                with db.get_connection() as conn:
                    cursor = conn.cursor()
                    
                    # Get table names
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = [row[0] for row in cursor.fetchall()]
                    
                    assert "screenshots" in tables
                    assert "ocr_results" in tables
                    assert "content_analysis" in tables
    
    def test_connection_context_manager(self):
        """Test database connection context manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_connection.db"
            
            with patch('eidolon.storage.metadata_db.get_config'), \
                 patch('eidolon.storage.metadata_db.get_component_logger'):
                
                db = MetadataDatabase(str(db_path))
                
                # Test connection context manager
                with db.get_connection() as conn:
                    assert conn is not None
                    assert isinstance(conn, sqlite3.Connection)
                    
                    # Test query works
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    assert result[0] == 1


class TestScreenshotOperations:
    """Test screenshot metadata operations."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_screenshots.db"
            
            with patch('eidolon.storage.metadata_db.get_config'), \
                 patch('eidolon.storage.metadata_db.get_component_logger'):
                
                yield MetadataDatabase(str(db_path))
    
    def test_store_screenshot_metadata(self, temp_db):
        """Test storing screenshot metadata."""
        screenshot_data = {
            "hash": "test_hash_123",
            "file_path": "/path/to/screenshot.png",
            "timestamp": datetime.now(),
            "size": (1920, 1080),
            "size_bytes": 524288,
            "window_info": {"title": "Test Window", "app": "TestApp"},
            "monitor_info": {"name": "Monitor 1", "primary": True}
        }
        
        screenshot_id = temp_db.store_screenshot(screenshot_data)
        
        assert screenshot_id is not None
        assert isinstance(screenshot_id, int)
        assert screenshot_id > 0
    
    def test_store_screenshot_duplicate_hash(self, temp_db):
        """Test handling duplicate screenshot hashes."""
        screenshot_data = {
            "hash": "duplicate_hash",
            "file_path": "/path/to/screenshot1.png",
            "timestamp": datetime.now(),
            "size": (1920, 1080)
        }
        
        # Store first screenshot
        id1 = temp_db.store_screenshot(screenshot_data)
        assert id1 is not None
        
        # Try to store duplicate hash - it uses INSERT OR REPLACE
        screenshot_data["file_path"] = "/path/to/screenshot2.png"
        id2 = temp_db.store_screenshot(screenshot_data)
        
        # Should return a valid ID (may be same or different)
        assert id2 is not None
    
    def test_search_screenshots_by_date_range(self, temp_db):
        """Test searching screenshots by date range."""
        now = datetime.now()
        old_time = now - timedelta(hours=2)
        recent_time = now - timedelta(minutes=30)
        
        # Store screenshots at different times
        id1 = temp_db.store_screenshot({
            "hash": "old_screenshot",
            "file_path": "/old.png",
            "timestamp": old_time,
            "size": (100, 100)
        })
        
        id2 = temp_db.store_screenshot({
            "hash": "recent_screenshot", 
            "file_path": "/recent.png",
            "timestamp": recent_time,
            "size": (100, 100)
        })
        
        assert id1 is not None
        assert id2 is not None
        
        # Search for recent screenshots
        start_time = now - timedelta(hours=1)
        end_time = now
        
        results = temp_db.get_screenshots_by_timerange(start_time, end_time)
        
        # The query might return screenshots even without content analysis
        # So we should check if any result has the recent hash
        assert len(results) >= 1
        assert any(r["hash"] == "recent_screenshot" for r in results)


class TestOCROperations:
    """Test OCR results storage."""
    
    @pytest.fixture
    def temp_db_with_screenshot(self):
        """Create database with a sample screenshot."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_ocr.db"
            
            with patch('eidolon.storage.metadata_db.get_config'), \
                 patch('eidolon.storage.metadata_db.get_component_logger'):
                
                db = MetadataDatabase(str(db_path))
                
                # Store a sample screenshot
                screenshot_id = db.store_screenshot({
                    "hash": "ocr_test_hash",
                    "file_path": "/ocr_test.png",
                    "timestamp": datetime.now(),
                    "size": (1024, 768)
                })
                
                yield db, screenshot_id
    
    def test_store_ocr_result(self, temp_db_with_screenshot):
        """Test storing OCR results."""
        db, screenshot_id = temp_db_with_screenshot
        
        ocr_result = {
            "text": "Hello World! This is test text.",
            "confidence": 0.95,
            "language": "en",
            "method": "tesseract",
            "word_count": 6,
            "regions": [
                {"text": "Hello World!", "bbox": [10, 20, 100, 40], "confidence": 0.98},
                {"text": "This is test text.", "bbox": [10, 50, 150, 70], "confidence": 0.92}
            ]
        }
        
        ocr_id = db.store_ocr_result(screenshot_id, ocr_result)
        
        assert ocr_id is not None
        assert isinstance(ocr_id, int)
        assert ocr_id > 0
    
    def test_search_ocr_text(self, temp_db_with_screenshot):
        """Test full-text search in OCR results."""
        db, screenshot_id = temp_db_with_screenshot
        
        # Store OCR results with searchable text
        ocr_results = [
            {"text": "Python programming tutorial", "confidence": 0.9, "language": "en", "method": "tesseract", "word_count": 3},
            {"text": "JavaScript development guide", "confidence": 0.8, "language": "en", "method": "tesseract", "word_count": 3},
            {"text": "Machine learning with Python", "confidence": 0.95, "language": "en", "method": "tesseract", "word_count": 4}
        ]
        
        for ocr_result in ocr_results:
            db.store_ocr_result(screenshot_id, ocr_result)
        
        # Search for Python-related content
        results = db.search_text("Python")
        
        # Should find text containing Python
        assert len(results) >= 2  # Should find both Python entries
        assert any("Python" in result["text"] for result in results)


class TestContentAnalysisOperations:
    """Test content analysis storage."""
    
    @pytest.fixture
    def temp_db_with_screenshot(self):
        """Create database with a sample screenshot."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_content.db"
            
            with patch('eidolon.storage.metadata_db.get_config'), \
                 patch('eidolon.storage.metadata_db.get_component_logger'):
                
                db = MetadataDatabase(str(db_path))
                
                screenshot_id = db.store_screenshot({
                    "hash": "content_test_hash",
                    "file_path": "/content_test.png", 
                    "timestamp": datetime.now(),
                    "size": (1024, 768)
                })
                
                yield db, screenshot_id
    
    def test_store_content_analysis(self, temp_db_with_screenshot):
        """Test storing content analysis results."""
        db, screenshot_id = temp_db_with_screenshot
        
        analysis = {
            "content_type": "code",
            "description": "Python programming code with functions",
            "confidence": 0.92,
            "tags": ["python", "programming", "code", "functions"],
            "ui_elements": [
                {"type": "editor", "text": "VS Code", "position": [0, 0]},
                {"type": "button", "text": "Run", "position": [100, 20]}
            ],
            "metadata": {
                "lines_of_code": 45,
                "language": "python",
                "framework": "django"
            }
        }
        
        analysis_id = db.store_content_analysis(screenshot_id, analysis)
        
        assert analysis_id is not None
        assert isinstance(analysis_id, int)
        assert analysis_id > 0
    
    def test_search_by_content_type(self, temp_db_with_screenshot):
        """Test searching content by type."""
        db, screenshot_id = temp_db_with_screenshot
        
        # Store different content types
        analyses = [
            {"content_type": "code", "description": "Code editor", "confidence": 0.9},
            {"content_type": "browser", "description": "Web browser", "confidence": 0.8},
            {"content_type": "code", "description": "Terminal output", "confidence": 0.85}
        ]
        
        for analysis in analyses:
            db.store_content_analysis(screenshot_id, analysis)
        
        # Search for code content
        results = db.get_content_by_type("code")
        
        assert len(results) == 2
        assert all(result["content_type"] == "code" for result in results)


class TestStatisticsAndMaintenance:
    """Test database statistics and maintenance operations."""
    
    @pytest.fixture  
    def populated_db(self):
        """Create database with sample data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_stats.db"
            
            with patch('eidolon.storage.metadata_db.get_config'), \
                 patch('eidolon.storage.metadata_db.get_component_logger'):
                
                db = MetadataDatabase(str(db_path))
                
                # Add sample data
                now = datetime.now()
                for i in range(5):
                    screenshot_id = db.store_screenshot({
                        "hash": f"hash_{i}",
                        "file_path": f"/screenshot_{i}.png",
                        "timestamp": now - timedelta(hours=i),
                        "size": (1024, 768)
                    })
                    
                    # Add OCR result
                    db.store_ocr_result(screenshot_id, {
                        "text": f"Sample text {i}",
                        "confidence": 0.9,
                        "method": "tesseract",
                        "language": "en",
                        "word_count": 2
                    })
                    
                    # Add content analysis
                    db.store_content_analysis(screenshot_id, {
                        "content_type": "document",
                        "description": f"Document {i}",
                        "confidence": 0.8
                    })
                
                yield db
    
    def test_get_database_statistics(self, populated_db):
        """Test retrieving database statistics."""
        stats = populated_db.get_statistics()
        
        assert isinstance(stats, dict)
        assert "total_screenshots" in stats
        assert "total_ocr_results" in stats
        assert "total_content_analyses" in stats
        assert stats["total_screenshots"] == 5
        assert stats["total_ocr_results"] == 5  
        assert stats["total_content_analyses"] == 5
    
    def test_cleanup_old_data(self, populated_db):
        """Test cleaning up old data."""
        # Clean up data older than 2 hours
        removed_count = populated_db.cleanup_old_data(days_to_keep=0.083)  # ~2 hours
        
        # Should remove some old data
        assert removed_count >= 0
        
        # Verify some data remains
        stats = populated_db.get_statistics()
        assert stats["total_screenshots"] > 0


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_database_path(self):
        """Test handling invalid database path."""
        with patch('eidolon.storage.metadata_db.get_config'), \
             patch('eidolon.storage.metadata_db.get_component_logger'):
            
            # Try to create database in non-existent directory (should create it)
            db = MetadataDatabase("/tmp/eidolon_test/nonexistent/test.db")
            assert db.db_path.parent.exists()
    
    def test_store_screenshot_missing_required_fields(self):
        """Test storing screenshot with missing required fields."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_error.db"
            
            with patch('eidolon.storage.metadata_db.get_config'), \
                 patch('eidolon.storage.metadata_db.get_component_logger'):
                
                db = MetadataDatabase(str(db_path))
                
                # Try to store screenshot without required fields
                try:
                    result = db.store_screenshot({
                        "hash": "test_hash"
                        # Missing required fields like file_path, timestamp, size
                    })
                    # Should either return None or raise exception
                    assert False, "Should have raised an error"
                except (KeyError, sqlite3.IntegrityError, TypeError):
                    # Expected behavior for missing required fields
                    pass


class TestMetadataDatabaseIntegration:
    """Integration tests for MetadataDatabase with realistic scenarios."""
    
    @pytest.fixture
    def integration_db(self):
        """Create database for integration testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_integration.db"
            
            with patch('eidolon.storage.metadata_db.get_config'), \
                 patch('eidolon.storage.metadata_db.get_component_logger'):
                
                yield MetadataDatabase(str(db_path))
    
    def test_complete_screenshot_workflow(self, integration_db):
        """Test complete workflow from screenshot to analysis."""
        # 1. Store screenshot
        screenshot_id = integration_db.store_screenshot({
            "hash": "workflow_test_hash",
            "file_path": "/test/workflow.png", 
            "timestamp": datetime.now(),
            "size": (1920, 1080),
            "window_info": {"title": "Test App", "pid": 12345}
        })
        
        assert screenshot_id is not None
        
        # 2. Store OCR result
        ocr_id = integration_db.store_ocr_result(screenshot_id, {
            "text": "def hello_world():\n    print('Hello, World!')",
            "confidence": 0.92,
            "language": "en",
            "method": "tesseract",
            "word_count": 4
        })
        
        assert ocr_id is not None
        
        # 3. Store content analysis
        analysis_id = integration_db.store_content_analysis(screenshot_id, {
            "content_type": "code",
            "description": "Python function definition",
            "confidence": 0.95,
            "tags": ["python", "function", "hello_world"],
            "metadata": {"language": "python", "function_count": 1}
        })
        
        assert analysis_id is not None
        
        # 4. Verify data can be searched
        text_results = integration_db.search_text("hello_world")
        assert len(text_results) > 0
        
        content_results = integration_db.get_content_by_type("code")
        assert len(content_results) > 0
    
    def test_bulk_operations_performance(self, integration_db):
        """Test performance with bulk operations."""
        import time
        
        start_time = time.time()
        
        # Store 10 screenshots with associated data
        for i in range(10):
            screenshot_id = integration_db.store_screenshot({
                "hash": f"bulk_test_hash_{i}",
                "file_path": f"/bulk/test_{i}.png",
                "timestamp": datetime.now(),
                "size": (1024, 768)
            })
            
            integration_db.store_ocr_result(screenshot_id, {
                "text": f"Bulk test content {i}",
                "confidence": 0.8 + (i * 0.01),
                "method": "tesseract",
                "language": "en",
                "word_count": 3
            })
            
            integration_db.store_content_analysis(screenshot_id, {
                "content_type": "document",
                "description": f"Bulk document {i}",
                "confidence": 0.9
            })
        
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 5.0  # 5 seconds for 10 complete workflows
        
        # Verify all data was stored
        stats = integration_db.get_statistics()
        assert stats["total_screenshots"] >= 10


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])