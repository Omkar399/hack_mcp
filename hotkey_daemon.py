#!/usr/bin/env python3
"""
Hotkey Daemon for Screen Memory Assistant
Runs in background and captures screenshots on keyboard shortcuts
"""
import asyncio
import logging
import os
import signal
import sys
from typing import Optional

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import keyboard
import httpx
from rich.console import Console
from rich.panel import Panel

# Setup logging and console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

# Configuration
API_BASE = os.getenv('SCREEN_MEMORY_API', 'http://localhost:5003')
CAPTURE_HOTKEY = 'cmd+shift+s'  # Default hotkey: Cmd+Shift+S (macOS)
FORCE_VISION_HOTKEY = 'cmd+shift+v'  # Force vision API: Cmd+Shift+V (macOS)


class HotkeyDaemon:
    """Background daemon for handling screenshot hotkeys"""
    
    def __init__(self, api_base: str = API_BASE):
        self.api_base = api_base
        self.client = httpx.AsyncClient(timeout=30.0)
        self.running = True
        
    async def capture_screenshot(self, force_vision: bool = False):
        """Capture screenshot via API"""
        try:
            console.print(f"📸 Capturing screenshot{'with vision analysis' if force_vision else ''}...")
            
            response = await self.client.post(f"{self.api_base}/capture_now", json={
                "save_image": True,
                "force_vision": force_vision
            })
            
            if response.status_code == 200:
                data = response.json()
                console.print(Panel.fit(f"""
✅ Screenshot captured successfully!

🪟 Window: {data.get('window_title', 'Unknown')[:50]}
📱 App: {data.get('app_name', 'Unknown')[:30]}
📄 Text: {len(data.get('full_text', ''))} characters
🔍 OCR Confidence: {data.get('ocr_conf', 0)}%
📸 Image: {data.get('image_path', 'Not saved')}
🤖 Vision Used: {'Yes' if force_vision or data.get('ocr_conf', 100) < 80 else 'No'}
                """, title="Capture Success", border_style="green"))
            else:
                console.print(f"❌ Capture failed: {response.status_code}", style="red")
                
        except Exception as e:
            console.print(f"❌ Capture error: {e}", style="red")
    
    def on_capture_hotkey(self):
        """Handle normal capture hotkey"""
        asyncio.create_task(self.capture_screenshot(force_vision=False))
    
    def on_vision_hotkey(self):
        """Handle force vision capture hotkey"""
        asyncio.create_task(self.capture_screenshot(force_vision=True))
    
    async def check_api_health(self):
        """Check if the API server is running"""
        try:
            response = await self.client.get(f"{self.api_base}/health")
            if response.status_code == 200:
                health_data = response.json()
                vision_available = health_data.get('capture_system', {}).get('vision_available', False)
                console.print(Panel.fit(f"""
🟢 API Server: Connected
🗄️  Database: {health_data.get('database', 'unknown')}
🔍 OCR: {', '.join(health_data.get('capture_system', {}).get('ocr_engines', ['none']))}
🤖 Vision API: {'Available' if vision_available else 'Disabled'}

Hotkeys:
• {CAPTURE_HOTKEY.upper()}: Normal screenshot
• {FORCE_VISION_HOTKEY.upper()}: Screenshot with AI analysis
• Ctrl+C: Exit daemon
                """, title="Screen Memory Hotkey Daemon", border_style="blue"))
                return True
            else:
                console.print(f"❌ API health check failed: {response.status_code}", style="red")
                return False
        except Exception as e:
            console.print(f"❌ Cannot connect to API server at {self.api_base}: {e}", style="red")
            return False
    
    def setup_hotkeys(self):
        """Setup keyboard hotkeys"""
        try:
            # Register hotkeys
            keyboard.add_hotkey(CAPTURE_HOTKEY, self.on_capture_hotkey)
            keyboard.add_hotkey(FORCE_VISION_HOTKEY, self.on_vision_hotkey)
            
            console.print(f"✅ Hotkeys registered:", style="green")
            console.print(f"   📸 {CAPTURE_HOTKEY.upper()}: Normal screenshot")
            console.print(f"   🤖 {FORCE_VISION_HOTKEY.upper()}: Screenshot with AI analysis")
            
        except Exception as e:
            console.print(f"❌ Failed to setup hotkeys: {e}", style="red")
            console.print("💡 Note: On macOS, you may need to grant accessibility permissions", style="yellow")
            return False
        
        return True
    
    def cleanup(self):
        """Cleanup resources"""
        keyboard.unhook_all()
        asyncio.create_task(self.client.aclose())
        console.print("🛑 Hotkey daemon stopped", style="yellow")
    
    async def run(self):
        """Main daemon loop"""
        console.print("🚀 Starting Screen Memory Hotkey Daemon...", style="bold blue")
        
        # Check API server
        if not await self.check_api_health():
            console.print("❌ Cannot start daemon - API server not available", style="red")
            console.print("💡 Start the API server first: uv run uvicorn screen_api:app --port 5003", style="yellow")
            return
        
        # Setup hotkeys
        if not self.setup_hotkeys():
            return
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            console.print("\n🛑 Received shutdown signal...", style="yellow")
            self.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        console.print("✅ Daemon running! Press Ctrl+C to stop.", style="green")
        
        # Keep daemon running
        try:
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()


async def main():
    """Main entry point"""
    daemon = HotkeyDaemon()
    await daemon.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye!", style="blue")
    except Exception as e:
        console.print(f"❌ Daemon crashed: {e}", style="red")
        sys.exit(1) 