#!/usr/bin/env python3
"""
Integration tests for Screen Memory Assistant
"""
import asyncio
import time
import httpx
import logging
from rich.console import Console
from rich.progress import track
from database import db
from capture import ScreenCapture

console = Console()
logger = logging.getLogger(__name__)

API_BASE = "http://localhost:5003"

async def test_database_connection():
    """Test database connectivity"""
    console.print("🔍 Testing database connection...", style="blue")
    
    try:
        await db.initialize()
        health = await db.health_check()
        
        if health:
            console.print("✅ Database connection successful", style="green")
            return True
        else:
            console.print("❌ Database health check failed", style="red")
            return False
    except Exception as e:
        console.print(f"❌ Database connection failed: {e}", style="red")
        return False


async def test_capture_system():
    """Test screen capture functionality"""
    console.print("📸 Testing capture system...", style="blue")
    
    try:
        capture_system = ScreenCapture()
        
        # Test basic capture
        capture_data = await capture_system.capture_screen(save_image=False)
        
        # Verify we got data
        assert capture_data.get('full_text') is not None, "No text extracted"
        assert isinstance(capture_data.get('ocr_conf'), int) or capture_data.get('ocr_conf') is None
        
        console.print("✅ Capture system working", style="green")
        console.print(f"  📄 Text length: {len(capture_data.get('full_text', ''))}")
        console.print(f"  🔍 OCR confidence: {capture_data.get('ocr_conf', 'N/A')}")
        console.print(f"  🖼️  CLIP embedding: {'Yes' if capture_data.get('clip_vec') else 'No'}")
        
        return True
        
    except Exception as e:
        console.print(f"❌ Capture system failed: {e}", style="red")
        return False


async def test_api_endpoints():
    """Test API endpoints"""
    console.print("🌐 Testing API endpoints...", style="blue")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        tests = [
            ("GET", "/health", "Health check"),
            ("GET", "/stats", "Statistics"),
            ("POST", "/capture_now", "Manual capture", {"save_image": False}),
            ("POST", "/find", "Search", {"query": "test", "limit": 5}),
            ("GET", "/recent", "Recent events"),
        ]
        
        results = []
        
        for method, endpoint, description, *args in tests:
            try:
                json_data = args[0] if args else None
                
                if method == "GET":
                    response = await client.get(f"{API_BASE}{endpoint}")
                else:
                    response = await client.post(f"{API_BASE}{endpoint}", json=json_data)
                
                if response.status_code < 400:
                    console.print(f"  ✅ {description} ({response.status_code})", style="green")
                    results.append(True)
                else:
                    console.print(f"  ❌ {description} ({response.status_code})", style="red")
                    results.append(False)
                    
            except httpx.ConnectError:
                console.print(f"  ❌ {description} (Connection failed - is server running?)", style="red")
                results.append(False)
            except Exception as e:
                console.print(f"  ❌ {description} (Error: {e})", style="red")
                results.append(False)
        
        success_rate = sum(results) / len(results)
        return success_rate > 0.8


async def test_search_functionality():
    """Test search functionality with real data"""
    console.print("🔍 Testing search functionality...", style="blue")
    
    try:
        # First capture something to search for
        capture_system = ScreenCapture()
        capture_data = await capture_system.capture_screen(save_image=False)
        
        # Save to database
        event_id = await db.save_screen_event(capture_data)
        
        # Wait a moment for database consistency
        await asyncio.sleep(1)
        
        # Test text search
        if capture_data.get('full_text'):
            # Search for a word from the captured text
            words = capture_data['full_text'].split()[:5]
            if words:
                search_word = words[0]
                results = await db.search_events(query=search_word, limit=5)
                
                if results:
                    console.print(f"  ✅ Text search found {len(results)} results for '{search_word}'", style="green")
                else:
                    console.print(f"  ⚠️  Text search found no results for '{search_word}'", style="yellow")
        
        # Test recent events
        recent = await db.get_recent_events(limit=10, hours=1)
        console.print(f"  ✅ Recent events: {len(recent)} found", style="green")
        
        return True
        
    except Exception as e:
        console.print(f"  ❌ Search functionality failed: {e}", style="red")
        return False


async def test_ml_components():
    """Test ML components availability"""
    console.print("🤖 Testing ML components...", style="blue")
    
    try:
        capture_system = ScreenCapture()
        health = capture_system.health_check()
        
        # Check OCR engines
        ocr_engines = health.get('ocr_engines', [])
        if ocr_engines:
            console.print(f"  ✅ OCR engines available: {', '.join(ocr_engines)}", style="green")
        else:
            console.print("  ❌ No OCR engines available", style="red")
        
        # Check CLIP
        clip_available = health.get('clip_available', False)
        if clip_available:
            console.print("  ✅ CLIP embeddings available", style="green")
        else:
            console.print("  ⚠️  CLIP embeddings not available", style="yellow")
        
        # Check Vision API
        vision_available = health.get('vision_available', False)
        if vision_available:
            console.print("  ✅ Vision API available", style="green")
        else:
            console.print("  ⚠️  Vision API not available", style="yellow")
        
        return len(ocr_engines) > 0
        
    except Exception as e:
        console.print(f"  ❌ ML components test failed: {e}", style="red")
        return False


async def test_performance():
    """Test system performance"""
    console.print("⚡ Testing performance...", style="blue")
    
    try:
        capture_system = ScreenCapture()
        
        # Time capture operations
        start_time = time.time()
        for i in range(3):
            await capture_system.capture_screen(save_image=False)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 3
        console.print(f"  📊 Average capture time: {avg_time:.2f}s", style="cyan")
        
        if avg_time < 5.0:
            console.print("  ✅ Performance acceptable", style="green")
            return True
        else:
            console.print("  ⚠️  Performance slower than expected", style="yellow")
            return False
            
    except Exception as e:
        console.print(f"  ❌ Performance test failed: {e}", style="red")
        return False


async def run_all_tests():
    """Run all integration tests"""
    console.print("🚀 Starting Screen Memory Assistant Integration Tests\n", style="bold blue")
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Capture System", test_capture_system),
        ("API Endpoints", test_api_endpoints),
        ("Search Functionality", test_search_functionality),
        ("ML Components", test_ml_components),
        ("Performance", test_performance),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        console.print(f"\n{'='*50}", style="dim")
        console.print(f"Running: {test_name}", style="bold")
        console.print(f"{'='*50}", style="dim")
        
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            console.print(f"❌ {test_name} crashed: {e}", style="red")
            results.append((test_name, False))
    
    # Summary
    console.print(f"\n{'='*60}", style="bold")
    console.print("🎯 TEST SUMMARY", style="bold blue")
    console.print(f"{'='*60}", style="bold")
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        console.print(f"{status} {test_name}", style="green" if success else "red")
        if success:
            passed += 1
    
    success_rate = passed / total
    console.print(f"\n📊 Overall: {passed}/{total} tests passed ({success_rate:.1%})", 
                 style="green" if success_rate > 0.8 else "yellow" if success_rate > 0.5 else "red")
    
    if success_rate > 0.8:
        console.print("🎉 System is ready for use!", style="green bold")
    elif success_rate > 0.5:
        console.print("⚠️  System has some issues but may work", style="yellow bold")
    else:
        console.print("❌ System has major issues", style="red bold")
    
    return success_rate


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    success_rate = asyncio.run(run_all_tests())
    
    # Exit with appropriate code
    exit(0 if success_rate > 0.8 else 1) 