#!/usr/bin/env python3
"""
Foreground App Capture for Screen Memory Assistant
Specifically designed to capture the active/foreground application
"""

import asyncio
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import systems
from database import db
from capture import ScreenCapture

async def capture_foreground():
    """Capture the foreground application screen"""
    
    print("📸 Capturing foreground app...")
    
    try:
        # Initialize database
        await db.initialize()
        print("✅ Database ready")
        
        # Initialize capture system
        capture_system = ScreenCapture()
        print("✅ Capture system ready")
        
        # Small delay to ensure proper window detection
        # This helps if the shortcut took focus briefly
        await asyncio.sleep(0.2)
        
        # Capture screen with current foreground app
        result = await capture_system.capture_screen(save_image=True)
        
        print(f"✅ Screenshot captured")
        print(f"📱 App: {result.get('app_name', 'Unknown')}")
        print(f"🪟 Window: {result.get('window_title', 'Unknown')}")
        
        # Store in database
        capture_data = {
            'app_name': result.get('app_name', 'Unknown App'),
            'window_title': result.get('window_title', 'Unknown Window'),
            'full_text': result.get('full_text', ''),
            'ocr_conf': result.get('ocr_conf', 0),
            'image_path': str(result.get('image_path', '')),
            'scene_hash': result.get('scene_hash', '')
        }
        
        event_id = await db.save_screen_event(capture_data)
        print(f"💾 Stored as event ID: {event_id}")
        
        # Show OCR preview
        if result.get('full_text'):
            preview = result['full_text'][:100] + "..." if len(result['full_text']) > 100 else result['full_text']
            print(f"📝 Text preview: {preview}")
        else:
            print("📝 No text detected")
        
        print("🎉 Capture complete!")
        return True
        
    except Exception as e:
        print(f"❌ Capture failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main entry point"""
    success = await capture_foreground()
    if not success:
        exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 