#!/usr/bin/env python3
"""
Development setup verification for Screen Memory Assistant
Tests that the system can run locally without Docker
"""
import asyncio
import sys
import os
from rich.console import Console
from rich.panel import Panel

console = Console()

async def test_basic_capture():
    """Test that we can do a basic screenshot and OCR"""
    console.print("🔍 Testing basic screen capture...", style="blue")
    
    try:
        from capture import ScreenCapture
        
        capture_system = ScreenCapture()
        
        # Test capture without saving image
        capture_data = await capture_system.capture_screen(save_image=False)
        
        if capture_data.get('full_text'):
            console.print(f"  ✅ OCR extracted {len(capture_data['full_text'])} characters", style="green")
        else:
            console.print("  ⚠️  OCR didn't extract any text (normal if screen is empty)", style="yellow")
        
        console.print(f"  📊 OCR confidence: {capture_data.get('ocr_conf', 'N/A')}%", style="cyan")
        console.print(f"  🖼️  CLIP embedding: {'Yes' if capture_data.get('clip_vec') else 'No'}", style="cyan")
        console.print(f"  🪟 Active window: {capture_data.get('window_title', 'Unknown')}", style="cyan")
        
        return True
        
    except Exception as e:
        console.print(f"  ❌ Basic capture test failed: {e}", style="red")
        return False

def test_api_imports():
    """Test that API components can be imported"""
    console.print("🌐 Testing API components...", style="blue")
    
    try:
        import screen_api
        from models import ScreenEventResponse, SearchRequest
        
        console.print("  ✅ FastAPI components imported successfully", style="green")
        
        # Test that we can create the app instance
        app = screen_api.app
        console.print("  ✅ FastAPI app instance created", style="green")
        
        return True
        
    except Exception as e:
        console.print(f"  ❌ API import test failed: {e}", style="red")
        return False

def test_cli_functionality():
    """Test CLI functionality"""
    console.print("🖱️  Testing CLI components...", style="blue")
    
    try:
        import cli
        
        # Test that CLI client can be created
        client = cli.ScreenMemoryClient("http://localhost:5003")
        
        console.print("  ✅ CLI client created successfully", style="green")
        
        return True
        
    except Exception as e:
        console.print(f"  ❌ CLI test failed: {e}", style="red")
        return False

async def main():
    """Main verification function"""
    console.print("🚀 Screen Memory Assistant - Development Setup Verification\n", style="bold blue")
    
    tests = [
        ("API Components", test_api_imports),
        ("CLI Functionality", test_cli_functionality), 
        ("Screen Capture", test_basic_capture),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        console.print(f"\n{'='*50}", style="dim")
        console.print(f"Running: {test_name}", style="bold")
        console.print(f"{'='*50}", style="dim")
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                success = await test_func()
            else:
                success = test_func()
            results.append((test_name, success))
        except Exception as e:
            console.print(f"❌ {test_name} crashed: {e}", style="red")
            results.append((test_name, False))
    
    # Summary
    console.print(f"\n{'='*60}", style="bold")
    console.print("🎯 VERIFICATION SUMMARY", style="bold blue")
    console.print(f"{'='*60}", style="bold")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    success_rate = passed / total
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        console.print(f"{status} {test_name}", style="green" if success else "red")
    
    console.print(f"\n📊 Overall: {passed}/{total} tests passed ({success_rate:.1%})", 
                 style="green" if success_rate > 0.8 else "yellow" if success_rate > 0.5 else "red")
    
    if success_rate > 0.8:
        console.print(Panel.fit("""
🎉 Development setup is working!

📋 Next steps:
1. Start the full Docker stack: ./start.sh
2. Or run in dev mode:
   - Start Postgres: docker run -p 5432:5432 -e POSTGRES_PASSWORD=hack123 -d postgres:16
   - Start Martian: docker run -p 5333:5333 -d ghcr.io/withmartian/router:latest
   - Start API: uv run uvicorn screen_api:app --reload --port 5003
   - Test CLI: uv run python cli.py health

🧪 Run full integration tests: uv run python test_integration.py
        """.strip(), title="Success!", border_style="green"))
    else:
        console.print(Panel.fit("""
❌ Development setup has issues

🔧 Try:
1. Reinstall dependencies: uv sync --extra dev --extra ml --extra ocr
2. Check that Python 3.11+ is installed
3. On macOS, install system dependencies: brew install tesseract
4. Check the error messages above for specific issues
        """.strip(), title="Issues Found", border_style="red"))
    
    return success_rate > 0.8

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        console.print("\n👋 Verification interrupted by user", style="yellow")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n💥 Unexpected error: {e}", style="red bold")
        sys.exit(1) 