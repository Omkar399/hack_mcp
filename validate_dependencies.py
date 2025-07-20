#!/usr/bin/env python3
"""
Eidolon AI Personal Assistant - Dependency Validation Script
Validates all dependencies across Phases 1-4 and checks for conflicts.
"""

import sys
import warnings
import importlib
from typing import List, Tuple

warnings.filterwarnings('ignore')

def test_import(module: str, description: str) -> Tuple[bool, str]:
    """Test import and return success status with version info."""
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, '__version__', 'unknown')
        return True, version
    except Exception as e:
        return False, str(e)[:80]

def main():
    """Run comprehensive dependency validation."""
    print("🔍 EIDOLON DEPENDENCY VALIDATION")
    print("=" * 50)
    
    # Define all critical dependencies by phase
    dependencies = {
        "Core System": [
            ("numpy", "NumPy"),
            ("PIL", "Pillow"),
            ("mss", "MSS (Screenshots)"),
            ("psutil", "PSUtil (System Monitoring)"),
            ("pydantic", "Pydantic"),
            ("yaml", "PyYAML"),
            ("dotenv", "Python-DotEnv"),
        ],
        "Phase 1 - Observer": [
            # Uses core dependencies only
        ],
        "Phase 2 - Analysis": [
            ("cv2", "OpenCV"),
            ("pytesseract", "PyTesseract"),
            ("easyocr", "EasyOCR"),
            ("skimage", "Scikit-Image"),
            ("scipy", "SciPy"),
            ("sklearn", "Scikit-Learn"),
        ],
        "Phase 3 - Local AI": [
            ("torch", "PyTorch"),
            ("torchvision", "TorchVision"),
            ("transformers", "Hugging Face Transformers"),
            ("timm", "TIMM"),
            ("einops", "Einops"),
            ("safetensors", "SafeTensors"),
        ],
        "Phase 4 - Cloud AI & Memory": [
            ("sentence_transformers", "Sentence Transformers"),
            ("chromadb", "ChromaDB"),
            ("aiohttp", "AioHTTP"),
            ("regex", "Regex"),
        ],
        "Cloud AI APIs (Optional)": [
            ("openai", "OpenAI"),
            ("anthropic", "Anthropic Claude"),
            ("google.generativeai", "Google Gemini"),
        ],
        "MCP Integration (v0.2.0)": [
            ("enrichmcp", "EnrichMCP Framework"),
            ("pyautogui", "PyAutoGUI"),
            ("pygetwindow", "PyGetWindow"),
            ("keyboard", "Keyboard Hooks"),
            ("structlog", "Structured Logging"),
            ("rich", "Rich Console"),
            ("requests", "HTTP Requests"),
        ]
    }
    
    total_tested = 0
    total_passed = 0
    
    for phase, deps in dependencies.items():
        if not deps:  # Skip empty phases
            continue
            
        print(f"\n{phase}:")
        phase_passed = 0
        
        for module, description in deps:
            success, info = test_import(module, description)
            status = "✅" if success else "❌"
            
            if success:
                print(f"  {status} {description}: v{info}")
                phase_passed += 1
                total_passed += 1
            else:
                print(f"  {status} {description}: {info}")
            
            total_tested += 1
        
        print(f"  Phase Status: {phase_passed}/{len(deps)} dependencies available")
    
    # Test Eidolon components
    print(f"\nEidolon Components:")
    components = [
        ("eidolon.core.observer", "Observer"),
        ("eidolon.core.analyzer", "Analyzer"),
        ("eidolon.storage.vector_db", "Vector Database"),
        ("eidolon.models.cloud_api", "Cloud API Manager"),
        ("eidolon.core.memory", "Memory System"),
        ("eidolon.mcp.server", "MCP Server (v0.2.0)"),
        ("eidolon.chat.chat_interface", "Chat Interface (v0.2.0)"),
    ]
    
    component_passed = 0
    for module, description in components:
        success, info = test_import(module, description)
        status = "✅" if success else "❌"
        print(f"  {status} {description}: {'OK' if success else info}")
        if success:
            component_passed += 1
    
    # Summary
    print(f"\n📊 VALIDATION SUMMARY")
    print("=" * 30)
    print(f"Dependencies: {total_passed}/{total_tested} ({total_passed/total_tested*100:.1f}%)")
    print(f"Components: {component_passed}/{len(components)} ({component_passed/len(components)*100:.1f}%)")
    
    # Check for known issues
    print(f"\n🔧 COMPATIBILITY CHECKS")
    print("-" * 25)
    
    # NumPy version check
    numpy_success, numpy_version = test_import("numpy", "NumPy")
    if numpy_success:
        try:
            major, minor = map(int, numpy_version.split('.')[:2])
            if major >= 2:
                print("⚠️  NumPy 2.x detected - some packages may need updates")
            else:
                print("✅ NumPy version compatible")
        except:
            print("⚠️  Could not parse NumPy version")
    
    # PyTorch CUDA check
    torch_success, _ = test_import("torch", "PyTorch")
    if torch_success:
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            print(f"{'✅' if cuda_available else 'ℹ️'} CUDA: {'Available' if cuda_available else 'Not available (CPU only)'}")
        except:
            print("ℹ️  Could not check CUDA availability")
    
    # Overall status
    overall_success = (total_passed == total_tested) and (component_passed == len(components))
    
    print(f"\n🎯 OVERALL STATUS")
    print("=" * 20)
    if overall_success:
        print("🎉 ALL SYSTEMS OPERATIONAL!")
        print("✅ Ready for production use")
        sys.exit(0)
    else:
        print("⚠️  Some dependencies missing")
        print("💡 Run: pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()