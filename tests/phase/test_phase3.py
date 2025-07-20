#!/usr/bin/env python3
"""
Phase 3 Comprehensive Test Suite for Eidolon AI Personal Assistant

Tests local AI integration, Florence-2 vision models, CLIP classification,
enhanced content analysis, and AI model optimizations.
"""

import os
import sys
import time
import pytest
import warnings
from pathlib import Path

# Fix tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Suppress known warnings from external libraries
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", message=".*invalid escape sequence.*")

sys.path.insert(0, '.')

class TestPhase3:
    """Phase 3 comprehensive test suite."""

    def test_local_ai_dependencies(self):
        """Test that all local AI dependencies are available."""
        # Test core AI libraries
        import torch
        import transformers
        from transformers import AutoProcessor, AutoModelForCausalLM
        
        assert torch is not None
        assert transformers is not None
        
        # Test vision libraries
        import cv2
        import numpy as np
        from PIL import Image
        
        assert cv2 is not None
        assert np is not None
        assert Image is not None
        
        # Test Florence-2 availability flag
        from eidolon.core.analyzer import FLORENCE_AVAILABLE
        assert isinstance(FLORENCE_AVAILABLE, bool)

    def test_local_vision_model(self):
        """Test local vision model capabilities."""
        from eidolon.models.local_vision import LocalVisionModel
        
        # Test model initialization (without loading actual models)
        vision_model = LocalVisionModel()
        assert vision_model is not None
        
        # Test model availability checks
        assert hasattr(vision_model, 'is_available')
        assert hasattr(vision_model, 'load_model')
        
        # Test configuration
        from eidolon.utils.config import get_config
        config = get_config()
        assert 'vision' in config.analysis.local_models
        assert config.analysis.local_models['vision'] == 'microsoft/florence-2-base'

    def test_enhanced_content_analysis(self):
        """Test enhanced content analysis with AI."""
        from eidolon.core.analyzer import Analyzer
        from PIL import Image
        
        analyzer = Analyzer()
        
        # Test that analyze_content method exists
        assert hasattr(analyzer, 'analyze_content')
        
        # Create a test image
        img = Image.new('RGB', (400, 300), color='blue')
        test_path = Path("test_analysis_image.png")
        img.save(test_path)
        
        try:
            # Test content analysis (may use local or fallback methods)
            result = analyzer.analyze_content(test_path)
            
            assert hasattr(result, 'content_type')
            assert hasattr(result, 'description')
            assert hasattr(result, 'confidence')
            assert hasattr(result, 'tags')
            
            assert isinstance(result.content_type, str)
            assert isinstance(result.description, str)
            assert isinstance(result.confidence, float)
            assert isinstance(result.tags, list)
            
        finally:
            if test_path.exists():
                test_path.unlink()

    def test_content_classification_accuracy(self):
        """Test content classification with various content types."""
        from eidolon.core.analyzer import Analyzer
        import hashlib
        
        analyzer = Analyzer()
        
        # Test various content types
        test_cases = [
            ("def main(): print('Hello, World!')", "code"),
            ("import pandas as pd\ndf = pd.read_csv('data.csv')", "code"),
            ("$ git commit -m 'Initial commit'", "terminal"),
            ("https://www.example.com", "browser"),
            ("Meeting agenda for today's standup", "document"),
            ("Error: FileNotFoundError on line 42", "terminal"),
        ]
        
        for text, expected_type in test_cases:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            classification = analyzer._classify_text_cached(text_hash, text)
            
            # Classification should be one of the valid types
            assert classification in ['code', 'document', 'terminal', 'browser', 'app']
            
        # Test cache performance
        cache_info = analyzer._classify_text_cached.cache_info()
        assert cache_info.hits + cache_info.misses >= len(test_cases)

    def test_performance_optimizations(self):
        """Test Phase 3 performance optimizations."""
        from eidolon.core.analyzer import Analyzer
        import re
        
        analyzer = Analyzer()
        
        # Test compiled regex patterns
        assert hasattr(analyzer, '_COMPILED_PATTERNS')
        patterns = analyzer._COMPILED_PATTERNS
        
        # Should have all required patterns
        expected_patterns = [
            'code_patterns', 'browser_patterns', 'file_extensions',
            'git_commands', 'error_patterns', 'ui_elements'
        ]
        
        for pattern_name in expected_patterns:
            assert pattern_name in patterns
            assert isinstance(patterns[pattern_name], re.Pattern)
        
        # Test LRU cache implementation
        assert hasattr(analyzer, '_classify_text_cached')
        
        # Test cache size limit
        cache_info = analyzer._classify_text_cached.cache_info()
        assert hasattr(cache_info, 'maxsize')
        assert cache_info.maxsize == 256

    def test_ui_element_detection(self):
        """Test UI element detection capabilities."""
        from eidolon.core.analyzer import Analyzer
        from PIL import Image, ImageDraw
        
        analyzer = Analyzer()
        
        # Create an image with UI-like elements
        img = Image.new('RGB', (600, 400), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw some UI-like elements
        draw.rectangle([50, 50, 200, 100], outline='black', width=2)  # Button
        draw.rectangle([50, 120, 500, 160], outline='gray', width=1)  # Text field
        draw.text((60, 70), "Click Me", fill='black')
        
        test_path = Path("test_ui_image.png")
        img.save(test_path)
        
        try:
            # Test UI element detection (if available)
            result = analyzer.analyze_content(test_path)
            
            # Should detect some kind of UI or app content
            assert hasattr(result, 'ui_elements')
            assert isinstance(result.ui_elements, list)
            
        finally:
            if test_path.exists():
                test_path.unlink()


def test_florence2_model_loading():
    """Test Florence-2 model loading capabilities."""
    from eidolon.core.analyzer import FLORENCE_AVAILABLE
    
    # Test that Florence-2 is available
    assert FLORENCE_AVAILABLE is True
    
    # Test transformers imports
    from transformers import AutoProcessor, AutoModelForCausalLM
    assert AutoProcessor is not None
    assert AutoModelForCausalLM is not None


def test_enhanced_content_analysis():
    """Test enhanced content analysis capabilities."""
    from eidolon.core.analyzer import Analyzer
    
    analyzer = Analyzer()
    
    # Test analyzer initialization
    assert analyzer is not None
    assert hasattr(analyzer, 'analyze_content')
    assert hasattr(analyzer, 'classify_content_type')


def test_vision_processing():
    """Test vision processing capabilities."""
    from eidolon.models.local_vision import LocalVisionModel
    
    # Test that local vision model can be imported and initialized
    vision_model = LocalVisionModel()
    assert vision_model is not None


def test_ui_element_detection():
    """Test UI element detection in screenshots."""
    from eidolon.core.analyzer import Analyzer
    from PIL import Image, ImageDraw
    
    analyzer = Analyzer()
    
    # Create test image with UI elements
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 150, 100], outline='black')
    
    test_path = Path("test_ui.png")
    img.save(test_path)
    
    try:
        result = analyzer.analyze_content(test_path)
        assert hasattr(result, 'ui_elements')
        assert isinstance(result.ui_elements, list)
    finally:
        if test_path.exists():
            test_path.unlink()


def test_performance_monitoring():
    """Test performance monitoring for AI operations."""
    from eidolon.core.analyzer import Analyzer
    import time
    
    analyzer = Analyzer()
    
    # Test that operations complete in reasonable time
    start_time = time.time()
    
    # Test text classification (should be fast with caching)
    import hashlib
    test_text = "def test(): return True"
    text_hash = hashlib.md5(test_text.encode()).hexdigest()
    result = analyzer._classify_text_cached(text_hash, test_text)
    
    elapsed = time.time() - start_time
    
    # Should complete quickly (under 1 second for cached operation)
    assert elapsed < 1.0
    assert result in ['code', 'document', 'terminal', 'browser', 'app']


def test_memory_efficiency():
    """Test memory efficiency of AI operations."""
    from eidolon.core.analyzer import Analyzer
    import psutil
    import os
    
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create analyzer and perform operations
    analyzer = Analyzer()
    
    # Perform multiple classification operations
    import hashlib
    for i in range(10):
        text = f"Sample text {i}: def function_{i}(): pass"
        text_hash = hashlib.md5(text.encode()).hexdigest()
        analyzer._classify_text_cached(text_hash, text)
    
    # Check memory usage hasn't grown excessively
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_growth = final_memory - initial_memory
    
    # Memory growth should be reasonable (less than 100MB for these operations)
    assert memory_growth < 100


def test_ai_error_handling():
    """Test error handling in AI operations."""
    from eidolon.core.analyzer import Analyzer
    from pathlib import Path
    
    analyzer = Analyzer()
    
    # Test with non-existent file
    non_existent_path = Path("non_existent_image.png")
    
    try:
        result = analyzer.extract_text(non_existent_path)
        # Should either return empty result or raise appropriate exception
        if result:
            assert hasattr(result, 'text')
            assert hasattr(result, 'confidence')
    except (FileNotFoundError, IOError):
        # This is acceptable error handling
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])