#!/usr/bin/env python3
"""
Test script for Eidolon MCP Integration

This script tests the basic functionality of the migrated EnrichMCP server
to ensure all components are working correctly.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_mcp_components():
    """Test individual MCP server components."""
    print("Testing Eidolon MCP Integration Components...")
    print("=" * 50)
    
    # Test 1: Import and basic initialization
    print("1. Testing imports and initialization...")
    try:
        from eidolon.core.mcp_server import (
            app, observer, analyzer, database, cloud_api,
            initialize_server, ScreenEvent, CaptureResult
        )
        from eidolon.core.observer import Observer
        from eidolon.core.analyzer import Analyzer
        from eidolon.storage.metadata_db import MetadataDatabase
        print("   ✓ All imports successful")
    except ImportError as e:
        print(f"   ✗ Import failed: {e}")
        return False
    
    # Test 2: Database initialization
    print("2. Testing database initialization...")
    try:
        # Use temporary database for testing
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            test_db = MetadataDatabase(tmp_db.name)
            stats = test_db.get_statistics()
            print(f"   ✓ Database initialized (screenshots: {stats['total_screenshots']})")
    except Exception as e:
        print(f"   ✗ Database initialization failed: {e}")
        return False
    
    # Test 3: Analyzer capabilities
    print("3. Testing analyzer capabilities...")
    try:
        test_analyzer = Analyzer()
        
        # Test error detection
        test_text = "Error: File not found\nTraceback (most recent call last):\n  File test.py"
        errors = test_analyzer.detect_errors(test_text)
        print(f"   ✓ Error detection: found {len(errors)} errors")
        
        # Test command extraction
        test_terminal = "$ git commit -m 'test'\n$ ls -la\n% npm install"
        commands = test_analyzer.extract_commands(test_terminal)
        print(f"   ✓ Command extraction: found {len(commands)} commands")
        
        # Test context analysis
        context = test_analyzer.analyze_context(test_terminal, "terminal")
        print(f"   ✓ Context analysis: activity type '{context['activity_type']}'")
        
    except Exception as e:
        print(f"   ✗ Analyzer testing failed: {e}")
        return False
    
    # Test 4: Cloud API manager (if available)
    print("4. Testing cloud API manager...")
    try:
        from eidolon.models.cloud_api import CloudAPIManager
        cloud_manager = CloudAPIManager()
        available_providers = cloud_manager.get_available_providers()
        print(f"   ✓ Cloud API manager initialized (providers: {available_providers})")
    except Exception as e:
        print(f"   ⚠ Cloud API manager: {e} (this is OK if no API keys are set)")
    
    # Test 5: MCP server models
    print("5. Testing MCP server models...")
    try:
        from datetime import datetime
        
        # Test ScreenEvent model
        screen_event = ScreenEvent(
            id=1,
            timestamp=datetime.now(),
            file_path="/test/path.png",
            hash="test123",
            full_text="test content",
            ocr_confidence=0.95,
            content_type="terminal",
            description="test description",
            tags=["test", "terminal"]
        )
        print(f"   ✓ ScreenEvent model: {screen_event.id}")
        
        # Test CaptureResult model
        capture_result = CaptureResult(
            screenshot_id=1,
            success=True,
            message="Test capture",
            processing_time=1.0,
            extracted_text="test text",
            content_type="terminal"
        )
        print(f"   ✓ CaptureResult model: {capture_result.success}")
        
    except Exception as e:
        print(f"   ✗ MCP model testing failed: {e}")
        return False
    
    # Test 6: EnrichMCP app configuration
    print("6. Testing EnrichMCP app...")
    try:
        print(f"   ✓ MCP app title: {app.title}")
        print(f"   ✓ MCP app description: {app.description[:50]}...")
    except Exception as e:
        print(f"   ✗ EnrichMCP app testing failed: {e}")
        return False
    
    print("\nAll component tests passed! ✓")
    return True

async def test_mcp_endpoints():
    """Test MCP endpoint functionality (without actual server)."""
    print("\n" + "=" * 50)
    print("Testing MCP Endpoint Functions...")
    print("=" * 50)
    
    try:
        # Import endpoint functions
        from eidolon.core.mcp_server import (
            get_system_status, search_screens, find_screens_with_errors,
            get_recent_screens
        )
        
        # Test system status
        print("1. Testing system status...")
        status = await get_system_status()
        print(f"   ✓ System status: monitoring={status.get('monitoring_active', False)}")
        
        # Test search (should handle empty database gracefully)
        print("2. Testing search functionality...")
        search_result = await search_screens("test query", limit=5)
        print(f"   ✓ Search: found {len(search_result.events)} results")
        
        # Test error finding
        print("3. Testing error detection...")
        errors = await find_screens_with_errors(severity="high", since_hours=24, limit=5)
        print(f"   ✓ Error detection: found {len(errors)} errors")
        
        # Test recent screens
        print("4. Testing recent screens...")
        recent = await get_recent_screens(limit=10, hours=24)
        print(f"   ✓ Recent screens: found {len(recent.items)} screens")
        
    except Exception as e:
        print(f"   ✗ Endpoint testing failed: {e}")
        return False
    
    print("\nAll endpoint tests passed! ✓")
    return True

def test_configuration():
    """Test configuration loading."""
    print("\n" + "=" * 50)
    print("Testing Configuration...")
    print("=" * 50)
    
    try:
        from eidolon.utils.config import get_config
        
        config = get_config()
        
        # Test MCP configuration
        if hasattr(config, 'mcp'):
            print(f"   ✓ MCP enabled: {config.mcp.enabled}")
            print(f"   ✓ MCP transport: {config.mcp.transport}")
            print(f"   ✓ MCP features: capture={config.mcp.enable_screen_capture}")
        else:
            print("   ⚠ MCP configuration not found (using defaults)")
        
        # Test other required configurations
        print(f"   ✓ Observer storage: {config.observer.storage_path}")
        print(f"   ✓ Database path: {config.memory.db_path}")
        
    except Exception as e:
        print(f"   ✗ Configuration testing failed: {e}")
        return False
    
    print("\nConfiguration test passed! ✓")
    return True

def test_cli_integration():
    """Test CLI integration."""
    print("\n" + "=" * 50)
    print("Testing CLI Integration...")
    print("=" * 50)
    
    try:
        from eidolon.cli.main import cli
        
        # Test that MCP command is available
        commands = [cmd.name for cmd in cli.commands.values()]
        if 'mcp' in commands:
            print("   ✓ MCP command registered in CLI")
        else:
            print("   ✗ MCP command not found in CLI")
            return False
        
        # Test help text
        from click.testing import CliRunner
        runner = CliRunner()
        result = runner.invoke(cli, ['mcp', '--help'])
        if result.exit_code == 0 and 'MCP' in result.output:
            print("   ✓ MCP command help available")
        else:
            print("   ⚠ MCP command help may have issues")
        
    except Exception as e:
        print(f"   ✗ CLI integration testing failed: {e}")
        return False
    
    print("\nCLI integration test passed! ✓")
    return True

async def main():
    """Run all tests."""
    print("Eidolon MCP Integration Test Suite")
    print("=" * 50)
    
    # Run all test suites
    tests_passed = 0
    total_tests = 4
    
    if await test_mcp_components():
        tests_passed += 1
    
    if await test_mcp_endpoints():
        tests_passed += 1
    
    if test_configuration():
        tests_passed += 1
    
    if test_cli_integration():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {tests_passed}/{total_tests} test suites passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! MCP integration is working correctly.")
        print("\nNext steps:")
        print("1. Install EnrichMCP: pip install enrichmcp>=0.4.5")
        print("2. Set up API keys for cloud AI (optional)")
        print("3. Start MCP server: python -m eidolon mcp")
        return True
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)