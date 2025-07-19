#!/usr/bin/env python3
"""
Quick verification test for Screen Memory Assistant
Tests imports and basic functionality without requiring Docker
"""
import sys
from rich.console import Console

console = Console()

def test_imports():
    """Test that all modules can be imported"""
    console.print("🔍 Testing imports...", style="blue")
    
    try:
        # Test core modules
        import models
        console.print("  ✅ models.py imports successfully", style="green")
        
        import database
        console.print("  ✅ database.py imports successfully", style="green")
        
        import capture
        console.print("  ✅ capture.py imports successfully", style="green")
        
        import screen_api
        console.print("  ✅ screen_api.py imports successfully", style="green")
        
        # Test CLI
        import cli
        console.print("  ✅ cli.py imports successfully", style="green")
        
        return True
        
    except ImportError as e:
        console.print(f"  ❌ Import failed: {e}", style="red")
        return False

def test_basic_functionality():
    """Test basic functionality without external dependencies"""
    console.print("🧪 Testing basic functionality...", style="blue")
    
    try:
        from capture import ScreenCapture
        
        # Test capture system initialization (without actually capturing)
        capture_system = ScreenCapture()
        health = capture_system.health_check()
        
        console.print(f"  ✅ Capture system initialized", style="green")
        console.print(f"  📊 OCR engines available: {len(health.get('ocr_engines', []))}", style="cyan")
        console.print(f"  🤖 CLIP available: {health.get('clip_available', False)}", style="cyan")
        console.print(f"  👁️  Vision available: {health.get('vision_available', False)}", style="cyan")
        
        return True
        
    except Exception as e:
        console.print(f"  ❌ Functionality test failed: {e}", style="red")
        return False

def test_models():
    """Test Pydantic models"""
    console.print("📋 Testing data models...", style="blue")
    
    try:
        from models import ScreenEventResponse, SearchRequest, CaptureRequest
        from datetime import datetime
        
        # Test model creation
        event = ScreenEventResponse(
            id=1,
            ts=datetime.now(),
            window_title="Test Window",
            app_name="Test App",
            full_text="Test text content",
            ocr_conf=85
        )
        
        search_req = SearchRequest(
            query="test query",
            limit=10
        )
        
        capture_req = CaptureRequest(
            save_image=True,
            force_vision=False
        )
        
        console.print("  ✅ All models create successfully", style="green")
        return True
        
    except Exception as e:
        console.print(f"  ❌ Model test failed: {e}", style="red")
        return False

def test_docker_files():
    """Test Docker configuration files exist and are readable"""
    console.print("🐳 Testing Docker configuration...", style="blue")
    
    try:
        import yaml
        
        # Test docker-compose.yml
        with open('docker-compose.yml', 'r') as f:
            compose_config = yaml.safe_load(f)
            
        services = compose_config.get('services', {})
        required_services = ['postgres', 'martian', 'app']
        
        for service in required_services:
            if service in services:
                console.print(f"  ✅ {service} service configured", style="green")
            else:
                console.print(f"  ❌ {service} service missing", style="red")
                return False
        
        # Test Dockerfile exists
        with open('Dockerfile', 'r') as f:
            dockerfile_content = f.read()
            
        if 'FROM python:3.11-slim' in dockerfile_content:
            console.print("  ✅ Dockerfile looks correct", style="green")
        else:
            console.print("  ❌ Dockerfile has issues", style="red")
            return False
        
        # Test requirements.txt
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
            
        key_packages = ['fastapi', 'sqlalchemy', 'torch', 'clip-by-openai', 'pytesseract']
        missing_packages = [pkg for pkg in key_packages if pkg not in requirements.lower()]
        
        if missing_packages:
            console.print(f"  ⚠️  Missing packages: {missing_packages}", style="yellow")
        else:
            console.print("  ✅ All key packages in requirements.txt", style="green")
        
        return True
        
    except Exception as e:
        console.print(f"  ❌ Docker config test failed: {e}", style="red")
        return False

def main():
    """Run all quick tests"""
    console.print("🚀 Screen Memory Assistant - Quick Verification\n", style="bold blue")
    
    tests = [
        ("Import Tests", test_imports),
        ("Basic Functionality", test_basic_functionality), 
        ("Data Models", test_models),
        ("Docker Configuration", test_docker_files),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        console.print(f"\n{'='*50}", style="dim")
        console.print(f"Running: {test_name}", style="bold")
        console.print(f"{'='*50}", style="dim")
        
        try:
            success = test_func()
            if success:
                passed += 1
        except Exception as e:
            console.print(f"❌ {test_name} crashed: {e}", style="red")
    
    # Summary
    console.print(f"\n{'='*60}", style="bold")
    console.print("🎯 VERIFICATION SUMMARY", style="bold blue")
    console.print(f"{'='*60}", style="bold")
    
    success_rate = passed / total
    console.print(f"📊 {passed}/{total} tests passed ({success_rate:.1%})", 
                 style="green" if success_rate == 1.0 else "yellow" if success_rate > 0.5 else "red")
    
    if success_rate == 1.0:
        console.print("🎉 System is ready to launch with Docker!", style="green bold")
        console.print("\n📋 Next steps:", style="bold")
        console.print("1. Run './start.sh' to launch the full system")
        console.print("2. Run 'python test_integration.py' for full end-to-end tests")
        console.print("3. Use 'python cli.py --help' to see available commands")
    else:
        console.print("❌ System has issues that need to be resolved", style="red bold")
    
    return success_rate == 1.0

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        console.print("\n👋 Test interrupted by user", style="yellow")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n💥 Unexpected error: {e}", style="red bold")
        sys.exit(1) 