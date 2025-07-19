#!/usr/bin/env python3
"""Direct test of the screen capture system"""
import asyncio
import os
from capture import ScreenCapture

async def test_capture():
    print("🔄 Testing direct screen capture...")
    
    try:
        # Initialize capture system
        capture_system = ScreenCapture()
        
        print("✅ Capture system initialized")
        
        # Test health
        health = capture_system.health_check()
        print(f"📊 Health: {health}")
        
        # Try a capture without saving image first
        print("🔄 Attempting screen capture (no image save)...")
        capture_data = await capture_system.capture_screen(save_image=False)
        
        print("✅ Capture successful!")
        print(f"📄 Text length: {len(capture_data.get('full_text', ''))}")
        print(f"🔍 OCR confidence: {capture_data.get('ocr_conf', 'N/A')}")
        print(f"🪟 Window title: {capture_data.get('window_title', 'N/A')}")
        print(f"📱 App name: {capture_data.get('app_name', 'N/A')}")
        print(f"🎯 Scene hash: {capture_data.get('scene_hash', 'N/A')[:20]}...")
        
        # Show first 200 characters of captured text
        if capture_data.get('full_text'):
            text_preview = capture_data['full_text'][:200]
            print(f"\n📝 Text preview: {text_preview}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Capture failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_capture())
    print(f"\n{'✅ Test passed!' if success else '❌ Test failed!'}") 