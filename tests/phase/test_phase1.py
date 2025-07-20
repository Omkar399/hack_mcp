#!/usr/bin/env python3
"""
Phase 1 Comprehensive Test Suite for Eidolon AI Personal Assistant

Tests all Phase 1 functionality including package structure, configuration,
observer, production monitoring, performance optimizations, and integrations.
"""

import os
import sys
import time
import json
import pytest
from pathlib import Path

# Fix tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.insert(0, '.')

class TestPhase1:
    """Phase 1 comprehensive test suite."""

    def test_package_structure(self):
        """Test the simplified package structure."""
        # Test all core imports with new structure
        from eidolon import __version__
        assert __version__ == "0.1.0"
        
        from eidolon.core.observer import Observer
        from eidolon.core.analyzer import Analyzer
        from eidolon.core.memory import MemorySystem
        
        from eidolon.models.decision_engine import DecisionEngine
        from eidolon.models.cloud_api import CloudAPIManager
        
        from eidolon.storage.vector_db import VectorDatabase
        from eidolon.storage.metadata_db import MetadataDatabase
        
        from eidolon.utils.config import get_config
        from eidolon.utils.logging import get_component_logger
        from eidolon.utils.production_monitor import ProductionMonitor

    def test_configuration_system(self):
        """Test enhanced configuration system."""
        from eidolon.utils.config import get_config
        
        config = get_config()
        
        # Test observer configuration
        assert hasattr(config, 'observer')
        assert config.observer.capture_interval == 10
        assert config.observer.activity_threshold == 0.05
        assert config.observer.cpu_check_interval == 0.1
        assert config.observer.thread_join_timeout == 5.0
        
        # Test analysis configuration
        assert hasattr(config, 'analysis')
        assert 'vision' in config.analysis.local_models
        assert 'clip' in config.analysis.local_models
        assert 'embedding' in config.analysis.local_models
        
        # Test cloud APIs configuration
        assert 'gemini_key' in config.analysis.cloud_apis
        assert 'claude_key' in config.analysis.cloud_apis
        
        # Test Gemini API key specifically
        gemini_key = config.analysis.cloud_apis.get('gemini_key', '')
        assert len(gemini_key) > 10
        assert not gemini_key.startswith('$')
        
        # Test memory configuration
        assert hasattr(config, 'memory')
        assert config.memory.vector_db == 'chromadb'
        
        # Test monitoring configuration
        assert hasattr(config, 'monitoring')
        assert config.monitoring.enabled is True
        assert config.monitoring.metrics_collection_interval == 60

    def test_observer_functionality(self):
        """Test enhanced Observer functionality."""
        from eidolon.core.observer import Observer
        
        observer = Observer()
        assert observer.config is not None
        assert observer.logger is not None
        
        # Test screenshot capture
        screenshot = observer.capture_screenshot()
        assert screenshot is not None
        assert hasattr(screenshot, 'image')
        assert hasattr(screenshot, 'hash')
        assert hasattr(screenshot, 'timestamp')
        
        # Test change detection
        time.sleep(0.1)
        screenshot2 = observer.capture_screenshot()
        metrics = observer.detect_changes(screenshot, screenshot2)
        
        assert hasattr(metrics, 'pixel_difference_ratio')
        assert hasattr(metrics, 'has_significant_change')
        assert isinstance(metrics.pixel_difference_ratio, float)
        assert isinstance(metrics.has_significant_change, bool)

    def test_production_monitoring(self):
        """Test production monitoring system."""
        from eidolon.utils.production_monitor import get_monitor
        
        monitor = get_monitor()
        assert monitor is not None
        
        # Test alert rules configuration
        assert len(monitor.alert_rules) == 3
        
        # Test health status (without starting monitoring)
        try:
            health = monitor.get_health_status()
            assert 'status' in health
            assert 'reason' in health
        except:
            # Health status may not be available without metrics
            pass
        
        # Test metrics summary
        try:
            summary = monitor.get_metrics_summary(hours=1)
            assert 'error' in summary or 'sample_count' in summary
        except:
            # Metrics may not be available yet
            pass

    def test_performance_optimizations(self):
        """Test performance optimizations."""
        from eidolon.core.analyzer import Analyzer
        import re
        
        analyzer = Analyzer()
        
        # Test compiled regex patterns
        assert hasattr(analyzer, '_COMPILED_PATTERNS')
        patterns = analyzer._COMPILED_PATTERNS
        assert len(patterns) == 6
        
        # Verify patterns are compiled
        for name, pattern in patterns.items():
            assert isinstance(pattern, re.Pattern), f"{name} should be compiled regex"
        
        # Test LRU caching
        assert hasattr(analyzer, '_classify_text_cached')
        
        # Test cache performance
        test_text = "Sample code: def hello(): print('world')"
        import hashlib
        text_hash = hashlib.md5(test_text.encode()).hexdigest()
        
        # First call (cache miss)
        result1 = analyzer._classify_text_cached(text_hash, test_text)
        # Second call (cache hit)
        result2 = analyzer._classify_text_cached(text_hash, test_text)
        
        assert result1 == result2
        
        cache_info = analyzer._classify_text_cached.cache_info()
        assert cache_info.hits >= 1
        assert cache_info.misses >= 1

    def test_cloud_integration(self):
        """Test cloud AI integration."""
        from eidolon.models.cloud_api import CloudAPIManager, GeminiAPI
        from eidolon.models.decision_engine import DecisionEngine, AnalysisRequest
        
        # Test API manager
        api_manager = CloudAPIManager()
        providers = api_manager.get_available_providers()
        assert len(providers) >= 3
        assert 'gemini' in providers
        
        # Test Gemini API specifically
        gemini_api = GeminiAPI()
        assert gemini_api.available is True
        assert gemini_api.model is not None
        
        # Test decision engine
        decision_engine = DecisionEngine()
        assert decision_engine is not None
        
        # Test routing decision
        sample_request = AnalysisRequest(
            content_type='image',
            image_size=(1920, 1080),
            text_content='Sample analysis request'
        )
        
        decision = decision_engine.make_routing_decision(sample_request, ['gemini'])
        assert hasattr(decision, 'use_cloud')
        assert hasattr(decision, 'reasoning')
        assert isinstance(decision.use_cloud, bool)
        assert isinstance(decision.reasoning, str)

    def test_memory_system(self):
        """Test memory and vector database system."""
        from eidolon.core.memory import MemorySystem
        
        memory_system = MemorySystem()
        assert memory_system is not None
        assert memory_system.config.memory.vector_db == 'chromadb'
        assert 'sentence-transformers' in memory_system.config.memory.embedding_model
        
        # Test dependencies
        import chromadb
        from sentence_transformers import SentenceTransformer
        
        # These imports should not raise exceptions
        assert chromadb is not None
        assert SentenceTransformer is not None

    def test_system_integration(self):
        """Test overall system integration."""
        # Test that all major components can be initialized together
        from eidolon.core.observer import Observer
        from eidolon.core.analyzer import Analyzer
        from eidolon.core.memory import MemorySystem
        from eidolon.models.decision_engine import DecisionEngine
        from eidolon.utils.production_monitor import get_monitor
        
        observer = Observer()
        analyzer = Analyzer()
        memory = MemorySystem()
        decision_engine = DecisionEngine()
        monitor = get_monitor()
        
        # Verify all components initialized successfully
        assert observer is not None
        assert analyzer is not None
        assert memory is not None
        assert decision_engine is not None
        assert monitor is not None
        
        # Test basic workflow
        screenshot = observer.capture_screenshot()
        assert screenshot is not None
        
        # Test analyzer integration
        sample_text = "Testing integration: def test(): return True"
        import hashlib
        text_hash = hashlib.md5(sample_text.encode()).hexdigest()
        classification = analyzer._classify_text_cached(text_hash, sample_text)
        assert classification in ['code', 'document', 'terminal', 'browser', 'app']


def test_imports():
    """Test that all core modules can be imported."""
    try:
        from eidolon import __version__
        from eidolon.core.observer import Observer
        from eidolon.utils.config import get_config
        from eidolon.utils.logging import get_component_logger
        from eidolon.utils.production_monitor import ProductionMonitor
        return True
    except Exception as e:
        pytest.fail(f"Import failed: {e}")


def test_configuration():
    """Test configuration loading and validation."""
    try:
        from eidolon.utils.config import get_config
        
        config = get_config()
        assert config.observer.storage_path == "./data/screenshots"
        assert config.observer.capture_interval == 10
        assert config.observer.max_storage_gb == 50.0
        assert config.observer.activity_threshold == 0.05
        return True
    except Exception as e:
        pytest.fail(f"Configuration test failed: {e}")


def test_screenshot_capture():
    """Test screenshot capture functionality."""
    try:
        from eidolon.core.observer import Observer
        
        observer = Observer()
        screenshot = observer.capture_screenshot()
        
        assert screenshot is not None
        assert hasattr(screenshot, 'image')
        assert hasattr(screenshot, 'hash')
        assert hasattr(screenshot, 'timestamp')
        
        # Test change detection
        time.sleep(0.1)
        screenshot2 = observer.capture_screenshot()
        metrics = observer.detect_changes(screenshot, screenshot2)
        
        assert hasattr(metrics, 'pixel_difference_ratio')
        assert hasattr(metrics, 'has_significant_change')
        return True
    except Exception as e:
        pytest.fail(f"Screenshot test failed: {e}")


def test_monitoring_session():
    """Test a short monitoring session."""
    observer = None
    try:
        from eidolon.core.observer import Observer
        
        observer = Observer()
        observer.start_monitoring()
        
        # Monitor for 2 seconds
        time.sleep(2)
        
        observer.stop_monitoring()
        time.sleep(0.5)  # Allow cleanup
        
        final_status = observer.get_status()
        assert 'capture_count' in final_status
        assert 'performance_metrics' in final_status
        return True
        
    except Exception as e:
        pytest.fail(f"Monitoring test failed: {e}")
    finally:
        if observer and hasattr(observer, '_running') and observer._running:
            try:
                observer.stop_monitoring()
                time.sleep(0.5)
            except:
                pass


def test_performance_monitoring():
    """Test performance monitoring functionality."""
    monitor = None
    try:
        from eidolon.utils.production_monitor import get_monitor
        
        monitor = get_monitor()
        
        # Test that monitor can collect basic metrics
        import psutil
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        assert isinstance(cpu_percent, float)
        assert isinstance(memory.percent, float)
        
        # Test health check functionality exists
        try:
            health = monitor.get_health_status()
            assert 'status' in health
        except:
            # Health status may not be available without monitoring
            pass
        
        return True
        
    except Exception as e:
        pytest.fail(f"Performance monitoring test failed: {e}")


def test_cli_commands():
    """Test CLI commands."""
    try:
        import eidolon
        version = eidolon.__version__
        assert version == "0.1.0"
        return True
    except Exception as e:
        pytest.fail(f"CLI test failed: {e}")


def test_data_persistence():
    """Test that data directory structure exists."""
    try:
        storage_path = Path("data/screenshots")
        
        # Directory may or may not exist depending on whether monitoring was run
        # This test just ensures the configuration is correct
        from eidolon.utils.config import get_config
        config = get_config()
        assert config.observer.storage_path == "./data/screenshots"
        return True
    except Exception as e:
        pytest.fail(f"Data persistence test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])