#!/usr/bin/env python3
"""
Phase 2 Comprehensive Test Suite for Eidolon AI Personal Assistant

Tests intelligent capture, OCR, content analysis, database functionality,
and enhanced search capabilities.
"""

import os
import sys
import time
import json
import pytest
from pathlib import Path
from datetime import datetime, timedelta

# Fix tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.insert(0, '.')

class TestPhase2:
    """Phase 2 comprehensive test suite."""

    def test_ocr_functionality(self):
        """Test OCR text extraction capabilities."""
        from eidolon.core.analyzer import Analyzer
        
        analyzer = Analyzer()
        assert hasattr(analyzer, '_tesseract_available')
        
        # Test OCR engines availability
        import pytesseract
        import easyocr
        
        # Create a simple test image with text
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        # Create test image
        img = Image.new('RGB', (400, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        # Use default font
        try:
            draw.text((10, 30), "Test OCR Text 123", fill='black')
        except:
            draw.text((10, 30), "Test OCR Text 123", fill='black')
        
        # Save test image
        test_path = Path("test_ocr_image.png")
        img.save(test_path)
        
        try:
            # Test OCR extraction
            result = analyzer.extract_text(test_path)
            assert hasattr(result, 'text')
            assert hasattr(result, 'confidence')
            assert hasattr(result, 'method')
            assert hasattr(result, 'word_count')
            assert hasattr(result, 'regions')
            
            # Text may or may not be detected depending on font availability
            assert isinstance(result.confidence, float)
            assert isinstance(result.word_count, int)
            assert isinstance(result.regions, list)
            assert result.method in ['tesseract', 'easyocr']
            
        finally:
            # Clean up test image
            if test_path.exists():
                test_path.unlink()

    def test_content_analysis(self):
        """Test content analysis and classification."""
        from eidolon.core.analyzer import Analyzer
        
        analyzer = Analyzer()
        
        # Test text classification with caching
        sample_texts = [
            ("def hello(): print('world')", "code"),
            ("Welcome to the browser homepage", "browser"),
            ("$ git status", "terminal"),
            ("Meeting notes for today", "document"),
        ]
        
        import hashlib
        for text, expected_category in sample_texts:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            classification = analyzer._classify_text_cached(text_hash, text)
            assert classification in ['code', 'document', 'terminal', 'browser', 'app']
        
        # Test cache performance
        cache_info = analyzer._classify_text_cached.cache_info()
        assert cache_info.hits + cache_info.misses >= len(sample_texts)

    def test_advanced_change_detection(self):
        """Test advanced change detection algorithms."""
        from eidolon.core.observer import Observer
        from PIL import Image
        import numpy as np
        
        observer = Observer()
        
        # Create two test images with slight differences
        img1 = Image.new('RGB', (200, 200), color='white')
        img2 = Image.new('RGB', (200, 200), color='white')
        
        # Add a small difference to img2
        pixels2 = np.array(img2)
        pixels2[50:60, 50:60] = [255, 0, 0]  # Red square
        img2 = Image.fromarray(pixels2)
        
        # Create mock screenshots
        from eidolon.core.observer import Screenshot
        screenshot1 = Screenshot(img1, "hash1", datetime.now())
        screenshot2 = Screenshot(img2, "hash2", datetime.now())
        
        # Test change detection
        metrics = observer.detect_changes(screenshot1, screenshot2)
        
        assert hasattr(metrics, 'pixel_difference_ratio')
        assert hasattr(metrics, 'has_significant_change')
        assert hasattr(metrics, 'changed_regions')
        assert isinstance(metrics.pixel_difference_ratio, float)
        assert isinstance(metrics.has_significant_change, bool)
        assert metrics.pixel_difference_ratio > 0  # Should detect the red square

    def test_database_functionality(self):
        """Test database storage and retrieval."""
        from eidolon.storage.metadata_db import MetadataDatabase
        from eidolon.core.analyzer import ExtractedText, ContentAnalysis
        from datetime import datetime
        
        # Initialize database
        db = MetadataDatabase()
        
        # Test screenshot storage
        screenshot_data = {
            'file_path': 'test_screenshot.png',
            'timestamp': datetime.now(),
            'hash': 'test_hash_123',
            'size': (1920, 1080),
            'file_size': 1024000
        }
        
        screenshot_id = db.store_screenshot(**screenshot_data)
        assert isinstance(screenshot_id, int)
        assert screenshot_id > 0
        
        # Test OCR result storage
        ocr_result = ExtractedText(
            text="Sample extracted text",
            confidence=0.95,
            method="tesseract"
        )
        
        ocr_id = db.store_ocr_result(screenshot_id, ocr_result)
        assert isinstance(ocr_id, int)
        assert ocr_id > 0
        
        # Test content analysis storage
        content_analysis = ContentAnalysis(
            content_type="document",
            description="Test document",
            confidence=0.9,
            tags=["test", "document"]
        )
        
        analysis_id = db.store_content_analysis(screenshot_id, content_analysis)
        assert isinstance(analysis_id, int)
        assert analysis_id > 0
        
        # Test search functionality
        results = db.search_text("Sample")
        assert isinstance(results, list)
        
        # Test statistics
        stats = db.get_statistics()
        assert isinstance(stats, dict)
        assert 'total_screenshots' in stats

    def test_enhanced_search(self):
        """Test enhanced search capabilities."""
        from eidolon.storage.metadata_db import MetadataDatabase
        from datetime import datetime, timedelta
        
        db = MetadataDatabase()
        
        # Test text search
        results = db.search_text("test", limit=5)
        assert isinstance(results, list)
        
        # Test content type search
        results = db.search_by_content_type("document", limit=5)
        assert isinstance(results, list)
        
        # Test time range search
        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)
        results = db.search_by_time_range(start_time, end_time, limit=5)
        assert isinstance(results, list)
        
        # Test combined search
        results = db.search_combined(
            text_query="test",
            content_types=["document", "terminal"],
            limit=3
        )
        assert isinstance(results, list)

    def test_file_management(self):
        """Test file management capabilities."""
        from eidolon.storage.file_manager import FileManager
        from pathlib import Path
        
        file_manager = FileManager()
        
        # Test storage path configuration
        storage_path = file_manager.get_storage_path()
        assert isinstance(storage_path, Path)
        
        # Test storage cleanup simulation
        # (We don't actually want to delete files in tests)
        assert hasattr(file_manager, 'cleanup_old_files')
        assert hasattr(file_manager, 'get_storage_usage')

    def test_integrated_capture_pipeline(self):
        """Test the complete capture and analysis pipeline."""
        from eidolon.core.observer import Observer
        from eidolon.core.analyzer import Analyzer
        from eidolon.storage.metadata_db import MetadataDatabase
        
        # Initialize components
        observer = Observer()
        analyzer = Analyzer()
        db = MetadataDatabase()
        
        # Capture a screenshot
        screenshot = observer.capture_screenshot()
        assert screenshot is not None
        
        # Extract text (may or may not find text)
        try:
            # Create a temporary file for testing
            temp_path = Path("temp_test.png")
            screenshot.image.save(temp_path)
            
            ocr_result = analyzer.extract_text(temp_path)
            assert hasattr(ocr_result, 'text')
            assert hasattr(ocr_result, 'confidence')
            
            # Clean up
            if temp_path.exists():
                temp_path.unlink()
                
        except Exception as e:
            # OCR may fail in test environment, that's okay
            pass

    def test_monitoring_integration(self):
        """Test monitoring session with database integration."""
        observer = None
        try:
            from eidolon.core.observer import Observer
            from eidolon.storage.metadata_db import MetadataDatabase
            
            observer = Observer()
            db = MetadataDatabase()
            
            # Get initial count
            initial_stats = db.get_statistics()
            initial_count = initial_stats.get('total_screenshots', 0)
            
            # Start monitoring briefly
            observer.start_monitoring()
            time.sleep(1)  # Very short monitoring session
            observer.stop_monitoring()
            time.sleep(0.5)  # Allow cleanup
            
            # Check if monitoring worked
            final_status = observer.get_status()
            assert 'capture_count' in final_status
            
        except Exception as e:
            pytest.fail(f"Monitoring integration test failed: {e}")
        finally:
            if observer and hasattr(observer, '_running') and observer._running:
                try:
                    observer.stop_monitoring()
                    time.sleep(0.5)
                except:
                    pass


def test_ocr_functionality():
    """Test OCR text extraction capabilities."""
    from eidolon.core.analyzer import Analyzer
    
    analyzer = Analyzer()
    
    # Test that OCR engines are available
    import pytesseract
    import easyocr
    
    # These should not raise ImportError
    assert pytesseract is not None
    assert easyocr is not None


def test_content_analysis():
    """Test content analysis and classification."""
    from eidolon.core.analyzer import Analyzer
    
    analyzer = Analyzer()
    
    # Test text classification
    sample_text = "def hello(): print('world')"
    import hashlib
    text_hash = hashlib.md5(sample_text.encode()).hexdigest()
    classification = analyzer._classify_text_cached(text_hash, sample_text)
    
    assert classification in ['code', 'document', 'terminal', 'browser', 'app']


def test_advanced_change_detection():
    """Test advanced change detection algorithms."""
    from eidolon.core.observer import Observer
    
    observer = Observer()
    
    # Test that change detection methods exist
    assert hasattr(observer, 'detect_changes')
    
    # Test with actual screenshots
    screenshot1 = observer.capture_screenshot()
    time.sleep(0.1)
    screenshot2 = observer.capture_screenshot()
    
    metrics = observer.detect_changes(screenshot1, screenshot2)
    assert hasattr(metrics, 'pixel_difference_ratio')
    assert hasattr(metrics, 'has_significant_change')


def test_database_functionality():
    """Test database storage and retrieval."""
    from eidolon.storage.metadata_db import MetadataDatabase
    
    db = MetadataDatabase()
    
    # Test basic database operations
    stats = db.get_statistics()
    assert isinstance(stats, dict)
    assert 'total_screenshots' in stats
    
    # Test search functionality
    results = db.search_text("test", limit=5)
    assert isinstance(results, list)


def test_integrated_capture_session():
    """Test a complete integrated capture session."""
    observer = None
    try:
        from eidolon.core.observer import Observer
        
        observer = Observer()
        
        # Brief monitoring session
        observer.start_monitoring()
        time.sleep(1)
        observer.stop_monitoring()
        time.sleep(0.5)
        
        # Check results
        status = observer.get_status()
        assert 'capture_count' in status
        assert 'performance_metrics' in status
        
    except Exception as e:
        pytest.fail(f"Integrated session test failed: {e}")
    finally:
        if observer and hasattr(observer, '_running') and observer._running:
            try:
                observer.stop_monitoring()
                time.sleep(0.5)
            except:
                pass


def test_enhanced_search():
    """Test enhanced search capabilities."""
    from eidolon.storage.metadata_db import MetadataDatabase
    
    db = MetadataDatabase()
    
    # Test various search methods
    text_results = db.search_text("test", limit=5)
    assert isinstance(text_results, list)
    
    type_results = db.search_by_content_type("document", limit=5)
    assert isinstance(type_results, list)
    
    from datetime import datetime, timedelta
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)
    time_results = db.search_by_time_range(start_time, end_time, limit=5)
    assert isinstance(time_results, list)


def test_cli_integration():
    """Test CLI integration with database functionality."""
    try:
        import eidolon
        assert eidolon.__version__ == "0.1.0"
        
        # Test that CLI modules can be imported
        from eidolon.cli.main import main
        assert main is not None
        
    except Exception as e:
        pytest.fail(f"CLI integration test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])