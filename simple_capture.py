#!/usr/bin/env python3
"""
Simple Screen Capture for Screen Memory Assistant
Can be easily bound to macOS shortcuts or run from terminal
"""
import asyncio
import sys
import subprocess
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import httpx
from rich.console import Console
from rich.panel import Panel

console = Console()


def show_notification(title: str, message: str, duration: float = 1.5):
    """Show macOS notification that auto-dismisses"""
    try:
        # Use display notification instead of alert (auto-dismisses)
        notification_script = f'''
        display notification "{message}" with title "📸 {title}"
        '''
        subprocess.run(['osascript', '-e', notification_script], check=False, capture_output=True)
        
    except Exception:
        pass  # Fail silently if notification fails


async def capture_screenshot(force_vision: bool = False, show_notification_flag: bool = True):
    """Capture screenshot via API"""
    try:
        # Show immediate feedback only if requested (avoid duplicate notifications)
        if show_notification_flag:
            show_notification("Capturing...", "Screenshot taken", duration=0.2)
        
        console.print(f"📸 Capturing screenshot{' with AI analysis' if force_vision else ''}...")
        
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post("http://localhost:5003/capture_now", json={
                "save_image": True,
                "force_vision": force_vision
            })
            
            if response.status_code == 200:
                data = response.json()
                
                # Create notification message
                window_title = data.get('window_title', 'Unknown')[:30]
                text_length = len(data.get('full_text', ''))
                vision_used = force_vision or data.get('ocr_conf', 100) < 80
                
                # Completion notification removed - only show initial "Capturing..." feedback
                
                # Also show console output for terminal use
                console.print(Panel.fit(f"""
✅ Screenshot captured successfully!

🪟 Window: {data.get('window_title', 'Unknown')[:50]}
📱 App: {data.get('app_name', 'Unknown')[:30]}  
📄 Text: {text_length} characters
🔍 OCR Confidence: {data.get('ocr_conf', 0)}%
📸 Image: {data.get('image_path', 'Not saved')}
🤖 Vision Used: {'Yes' if vision_used else 'No'}
⏰ Time: {datetime.now().strftime('%H:%M:%S')}
                """, title="Capture Success", border_style="green"))
                
                # Show a preview of captured text
                if data.get('full_text'):
                    preview = data.get('full_text', '')[:100]
                    console.print(f"📝 Text Preview: {preview}{'...' if len(data.get('full_text', '')) > 100 else ''}")
                
            else:
                console.print(f"❌ Capture failed: {response.status_code}", style="red")
                show_notification("Screen Memory Error", f"Capture failed: {response.status_code}")
                
    except httpx.ConnectError:
        console.print("❌ Cannot connect to API server. Is it running?", style="red")
        console.print("💡 Start it with: uv run uvicorn screen_api:app --port 5003", style="yellow")
        show_notification("Error", "Server not running", duration=2.0)
    except httpx.TimeoutException:
        console.print("⏱️ Capture timed out - server may be overloaded", style="yellow")
        show_notification("Timeout", "Processing took too long", duration=2.0)
    except Exception as e:
        console.print(f"❌ Capture error: {e}", style="red")
        show_notification("Error", "Capture failed", duration=2.0)


async def main():
    """Main entry point"""
    force_vision = '--vision' in sys.argv or '-v' in sys.argv
    no_notification = '--no-notification' in sys.argv or '--silent' in sys.argv
    
    if '--help' in sys.argv or '-h' in sys.argv:
        console.print(Panel.fit("""
🖥️  Simple Screen Capture

Usage:
  python simple_capture.py           # Normal screenshot
  python simple_capture.py --vision  # Force AI analysis
  python simple_capture.py -v        # Force AI analysis (short)
  
🔧 To use with macOS Shortcuts:
  1. Open Shortcuts app
  2. Create new shortcut
  3. Add "Run Shell Script" action
  4. Set script to: cd /path/to/hack_mcp && uv run python simple_capture.py
  5. Assign keyboard shortcut (e.g., Cmd+Shift+S)
        """, title="Help", border_style="blue"))
        return
    
    await capture_screenshot(force_vision=force_vision, show_notification_flag=not no_notification)


if __name__ == "__main__":
    asyncio.run(main()) 