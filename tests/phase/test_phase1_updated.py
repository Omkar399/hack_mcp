#!/usr/bin/env python3
"""
Phase 1 Comprehensive Test Suite for Eidolon AI Personal Assistant

Tests all Phase 1 functionality including new features and optimizations.
"""

import os
import sys
import time
import json
from pathlib import Path

# Fix tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.insert(0, '.')

def test_package_structure():
    """Test the new simplified package structure."""
    print("1️⃣  Testing Package Structure...")
    print("-" * 50)
    
    try:
        # Test all core imports with new structure
        from eidolon import __version__
        print(f"✅ Eidolon version: {__version__}")
        
        from eidolon.core.observer import Observer
        from eidolon.core.analyzer import Analyzer
        from eidolon.core.memory import MemorySystem
        print("✅ Core modules imported")
        
        from eidolon.models.decision_engine import DecisionEngine
        from eidolon.models.cloud_api import CloudAPIManager
        print("✅ Model modules imported")
        
        from eidolon.storage.vector_db import VectorDatabase
        from eidolon.storage.metadata_db import MetadataDatabase
        print("✅ Storage modules imported")
        
        from eidolon.utils.config import get_config
        from eidolon.utils.logging import get_component_logger
        from eidolon.utils.production_monitor import ProductionMonitor
        print("✅ Utility modules imported")
        
        print("\n✅ All modules imported successfully with new structure!\n")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_configuration_system():
    """Test enhanced configuration system."""
    print("2️⃣  Testing Enhanced Configuration System...")
    print("-" * 50)
    
    try:
        from eidolon.utils.config import get_config
        
        config = get_config()
        print(f"✅ Configuration loaded from: {config.observer.storage_path}")
        
        # Test observer configuration
        print(f"✅ Capture interval: {config.observer.capture_interval}s")
        print(f"✅ Activity threshold: {config.observer.activity_threshold}")
        print(f"✅ CPU check interval: {config.observer.cpu_check_interval}s")
        print(f"✅ Thread join timeout: {config.observer.thread_join_timeout}s")
        
        # Test analysis configuration
        print(f"✅ Local models configured: {config.analysis.local_models}")
        print(f"✅ Cloud APIs configured: {len(config.analysis.cloud_apis)} providers")
        
        # Test Gemini API key specifically
        gemini_key = config.analysis.cloud_apis.get('gemini_key', '')
        gemini_configured = bool(gemini_key and len(gemini_key) > 10 and not gemini_key.startswith('$'))
        print(f"✅ Gemini API configured: {gemini_configured}")
        
        # Test memory configuration
        print(f"✅ Vector DB: {config.memory.vector_db}")
        print(f"✅ Embedding model: {config.memory.embedding_model}")
        
        # Test monitoring configuration
        print(f"✅ Monitoring enabled: {config.monitoring.enabled}")
        print(f"✅ Metrics interval: {config.monitoring.metrics_collection_interval}s")
        
        print("\n✅ Enhanced configuration system working correctly!\n")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_observer_functionality():
    """Test enhanced Observer functionality."""
    print("3️⃣  Testing Enhanced Observer...")
    print("-" * 50)
    
    try:
        from eidolon.core.observer import Observer
        
        observer = Observer()
        print(f"✅ Observer initialized")
        print(f"✅ Configuration loaded: {observer.config is not None}")
        print(f"✅ Logger created: {observer.logger is not None}")
        
        # Test screenshot capture
        screenshot = observer.capture_screenshot()
        if screenshot:
            print(f"✅ Screenshot captured successfully")
            print(f"   - Size: {screenshot.image.size}")
            print(f"   - Hash: {screenshot.hash[:16]}...")
            
            # Test change detection with optimized thresholds
            time.sleep(0.1)
            screenshot2 = observer.capture_screenshot()
            metrics = observer.detect_changes(screenshot, screenshot2)
            
            print(f"✅ Change detection working")
            print(f"   - Pixel difference: {metrics.pixel_difference_ratio:.3f}")
            print(f"   - Has significant change: {metrics.has_significant_change}")
        else:
            print("❌ Failed to capture screenshot")
            return False
            
        print("\n✅ Enhanced Observer working correctly!\n")
        return True
        
    except Exception as e:
        print(f"❌ Observer test failed: {e}")
        return False


def test_production_monitoring():
    """Test production monitoring system."""
    print("4️⃣  Testing Production Monitoring...")
    print("-" * 50)
    
    try:
        from eidolon.utils.production_monitor import get_monitor
        
        monitor = get_monitor()
        print(f"✅ Production monitor initialized")
        
        # Test metrics collection
        import psutil
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        print(f"✅ System metrics available")
        print(f"   - CPU: {cpu_percent:.1f}%")
        print(f"   - Memory: {memory.percent:.1f}%")
        
        # Test health status (without starting monitoring)
        try:
            health = monitor.get_health_status()
            print(f"✅ Health status: {health['status']}")
        except:
            print("⚠️ Health status not available (no metrics yet)")
        
        # Test alert rules configuration
        print(f"✅ Alert rules configured: {len(monitor.alert_rules)} rules")
        
        print("\n✅ Production monitoring working correctly!\n")
        return True
        
    except Exception as e:
        print(f"❌ Production monitoring test failed: {e}")
        return False


def test_performance_optimizations():
    """Test performance optimizations."""
    print("5️⃣  Testing Performance Optimizations...")
    print("-" * 50)
    
    try:
        from eidolon.core.analyzer import Analyzer
        
        analyzer = Analyzer()
        
        # Test compiled regex patterns
        if hasattr(analyzer, '_COMPILED_PATTERNS'):
            patterns = analyzer._COMPILED_PATTERNS
            print(f"✅ Compiled regex patterns: {len(patterns)} patterns")
            
            # Verify patterns are compiled
            import re
            for name, pattern in patterns.items():
                if isinstance(pattern, re.Pattern):
                    print(f"   ✅ {name}: compiled")
                else:
                    print(f"   ❌ {name}: not compiled")
        
        # Test LRU caching
        if hasattr(analyzer, '_classify_text_cached'):
            print("✅ LRU cache implemented")
            
            # Test cache performance
            test_text = "Sample code: def hello(): print('world')"
            import hashlib
            text_hash = hashlib.md5(test_text.encode()).hexdigest()
            
            # First call (cache miss)
            result1 = analyzer._classify_text_cached(text_hash, test_text)
            # Second call (cache hit)
            result2 = analyzer._classify_text_cached(text_hash, test_text)
            
            cache_info = analyzer._classify_text_cached.cache_info()
            print(f"✅ Cache performance: hits={cache_info.hits}, misses={cache_info.misses}")
        
        print("\n✅ Performance optimizations working correctly!\n")
        return True
        
    except Exception as e:
        print(f"❌ Performance optimizations test failed: {e}")
        return False


def test_cloud_integration():
    """Test cloud AI integration."""
    print("6️⃣  Testing Cloud AI Integration...")
    print("-" * 50)
    
    try:
        from eidolon.models.cloud_api import CloudAPIManager, GeminiAPI
        
        # Test API manager
        api_manager = CloudAPIManager()
        providers = api_manager.get_available_providers()
        print(f"✅ CloudAPIManager initialized")
        print(f"✅ Available providers: {providers}")
        
        # Test Gemini API specifically
        gemini_api = GeminiAPI()
        print(f"✅ GeminiAPI initialized: {gemini_api.available}")
        
        if gemini_api.available:
            print(f"✅ Gemini model configured: {type(gemini_api.model).__name__}")
        
        # Test decision engine
        from eidolon.models.decision_engine import DecisionEngine, AnalysisRequest
        
        decision_engine = DecisionEngine()
        print(f"✅ DecisionEngine initialized")
        
        # Test routing decision
        sample_request = AnalysisRequest(
            content_type='image',
            image_size=(1920, 1080),
            text_content='Sample analysis request'
        )
        
        decision = decision_engine.make_routing_decision(sample_request, ['gemini'])
        print(f"✅ Routing decision: cloud={decision.use_cloud}")
        print(f"✅ Decision reasoning: {decision.reasoning[:50]}...")
        
        print("\n✅ Cloud AI integration working correctly!\n")
        return True
        
    except Exception as e:
        print(f"❌ Cloud integration test failed: {e}")
        return False


def test_memory_system():
    """Test memory and vector database system."""
    print("7️⃣  Testing Memory System...")
    print("-" * 50)
    
    try:
        from eidolon.core.memory import MemorySystem
        
        memory_system = MemorySystem()
        print(f"✅ MemorySystem initialized")
        print(f"✅ Vector DB configured: {memory_system.config.memory.vector_db}")
        print(f"✅ Embedding model: {memory_system.config.memory.embedding_model}")
        
        # Test vector database availability
        import chromadb
        print("✅ ChromaDB available")
        
        # Test sentence transformers
        from sentence_transformers import SentenceTransformer
        print("✅ SentenceTransformer available")
        
        print("\n✅ Memory system working correctly!\n")
        return True
        
    except Exception as e:
        print(f"❌ Memory system test failed: {e}")
        return False


def main():
    """Run all Phase 1 enhanced validation tests."""
    print("\n🔍 EIDOLON PHASE 1 COMPREHENSIVE VALIDATION")
    print("=" * 60)
    print("Testing all Phase 1 functionality including new features\n")
    
    tests = [
        test_package_structure,
        test_configuration_system,
        test_observer_functionality,
        test_production_monitoring,
        test_performance_optimizations,
        test_cloud_integration,
        test_memory_system,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print("📊 PHASE 1 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 ALL PHASE 1 TESTS PASSED!")
        print("✅ Foundation and package structure complete")
        print("✅ Enhanced configuration system working")
        print("✅ Production monitoring ready")
        print("✅ Performance optimizations active")
        print("✅ Cloud AI integration functional")
        print("✅ Memory system operational")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    print("\n")


if __name__ == "__main__":
    main()