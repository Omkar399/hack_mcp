"""
Screen Memory Chat Bot

A native macOS popup chat interface that queries screen memory using the MCP server.
"""

import os
import sys
import json
import asyncio
import logging
import subprocess
from datetime import datetime
from typing import List, Dict, Any, Optional
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import httpx

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if MCP server is running
MCP_SERVER_URL = "http://localhost:8000"  # Default MCP server port


class ScreenMemoryChatBot:
    """Native macOS chat interface for screen memory queries."""
    
    def __init__(self):
        self.root = None
        self.chat_display = None
        self.input_field = None
        self.send_button = None
        self.status_label = None
        self.conversation_history = []
        
        # HTTP client for MCP server
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Set up the UI
        self.setup_ui()
        
    def setup_ui(self):
        """Create the chat interface."""
        self.root = tk.Tk()
        self.root.title("Screen Memory Assistant")
        self.root.geometry("600x500")
        
        # Make window stay on top and center it
        self.root.wm_attributes("-topmost", True)
        self.center_window()
        
        # Configure style
        style = ttk.Style()
        style.theme_use('aqua')  # Use macOS native theme
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Screen Memory Assistant", 
                               font=("SF Pro Display", 16, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 10), sticky=tk.W)
        
        # Chat display area
        chat_frame = ttk.Frame(main_frame)
        chat_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=("SF Pro Text", 12),
            bg="#f8f9fa",
            fg="#212529"
        )
        self.chat_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Input frame
        input_frame = ttk.Frame(main_frame)
        input_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(0, weight=1)
        
        # Input field
        self.input_field = ttk.Entry(input_frame, font=("SF Pro Text", 12))
        self.input_field.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        self.input_field.bind('<Return>', self.on_send_message)
        self.input_field.bind('<Shift-Return>', lambda e: None)  # Allow newlines with Shift+Enter
        
        # Send button
        self.send_button = ttk.Button(input_frame, text="Send", command=self.on_send_message)
        self.send_button.grid(row=0, column=1)
        
        # Status bar
        self.status_label = ttk.Label(main_frame, text="Ready", foreground="green")
        self.status_label.grid(row=3, column=0, sticky=tk.W)
        
        # Add initial welcome message
        self.add_message("assistant", 
            "Hi! I'm your Screen Memory Assistant. I can help you find and analyze your screen captures.\n\n"
            "Try asking me:\n"
            "• 'What was I working on 10 minutes ago?'\n"
            "• 'Find login forms from today'\n"
            "• 'Show me recent error messages'\n"
            "• 'Capture my screen now'"
        )
        
        # Set focus to input field
        self.input_field.focus_set()
        
    def center_window(self):
        """Center the window on the screen."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        
    def add_message(self, sender: str, message: str):
        """Add a message to the chat display."""
        self.chat_display.config(state=tk.NORMAL)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%H:%M")
        
        if sender == "user":
            self.chat_display.insert(tk.END, f"[{timestamp}] You: {message}\n\n")
        else:
            self.chat_display.insert(tk.END, f"[{timestamp}] Assistant: {message}\n\n")
        
        # Auto-scroll to bottom
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
        
    def set_status(self, message: str, color: str = "black"):
        """Update the status bar."""
        self.status_label.config(text=message, foreground=color)
        self.root.update()
        
    def on_send_message(self, event=None):
        """Handle sending a message."""
        message = self.input_field.get().strip()
        if not message:
            return
            
        # Clear input field
        self.input_field.delete(0, tk.END)
        
        # Add user message to display
        self.add_message("user", message)
        
        # Process message in background thread
        threading.Thread(target=self.process_message, args=(message,), daemon=True).start()
        
    def process_message(self, message: str):
        """Process user message and get response."""
        try:
            self.set_status("Thinking...", "orange")
            
            # Run async processing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(self.get_ai_response(message))
            loop.close()
            
            # Add response to display
            self.add_message("assistant", response)
            self.set_status("Ready", "green")
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.add_message("assistant", f"Sorry, I encountered an error: {str(e)}")
            self.set_status("Error", "red")
            
    async def get_ai_response(self, message: str) -> str:
        """Get AI response using the MCP server."""
        try:
            # Check if this is a capture request
            if any(word in message.lower() for word in ["capture", "screenshot", "snap"]):
                return await self.handle_capture_request(message)
            
            # Check if this is a search request
            if any(word in message.lower() for word in ["find", "search", "show", "what", "when", "where"]):
                return await self.handle_search_request(message)
            
            # Default to context analysis
            return await self.handle_context_analysis(message)
            
        except httpx.ConnectError:
            return ("I couldn't connect to the Screen Memory server. "
                   "Please make sure it's running with: python mcp_server.py")
        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            return f"I encountered an error: {str(e)}"
            
    async def handle_capture_request(self, message: str) -> str:
        """Handle screen capture requests."""
        try:
            # Determine capture options from message
            use_vision = "vision" in message.lower() or "ai" in message.lower()
            
            response = await self.client.post(
                f"{MCP_SERVER_URL}/capture_screen",
                json={
                    "save_image": True,
                    "use_vision": use_vision
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                if result["success"]:
                    return (f"✅ Screen captured successfully!\n"
                           f"Event ID: {result['event_id']}\n"
                           f"Processing time: {result['processing_time']:.2f}s")
                else:
                    return f"❌ Capture failed: {result['message']}"
            else:
                return f"❌ Server error: {response.status_code}"
                
        except Exception as e:
            return f"❌ Capture error: {str(e)}"
            
    async def handle_search_request(self, message: str) -> str:
        """Handle search requests."""
        try:
            # Extract search parameters from message
            limit = 5  # Default limit for chat responses
            since_hours = None
            
            # Simple keyword extraction
            if "today" in message.lower():
                since_hours = 24
            elif "hour" in message.lower():
                since_hours = 1
            elif "recent" in message.lower():
                since_hours = 6
                
            response = await self.client.post(
                f"{MCP_SERVER_URL}/search_screens",
                json={
                    "query": message,
                    "limit": limit,
                    "since_hours": since_hours
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                events = result["events"]
                
                if not events:
                    return f"🔍 No screen captures found matching: '{message}'"
                
                response_text = f"🔍 Found {len(events)} relevant screen capture(s):\n\n"
                
                for i, event in enumerate(events, 1):
                    timestamp = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
                    app = event["app_name"] or "Unknown App"
                    window = event["window_title"] or ""
                    
                    response_text += f"{i}. [{timestamp.strftime('%H:%M')}] {app}"
                    if window:
                        response_text += f" - {window}"
                    
                    if event["full_text"]:
                        # Show first 100 characters of text
                        text_preview = event["full_text"][:100]
                        if len(event["full_text"]) > 100:
                            text_preview += "..."
                        response_text += f"\n   Text: {text_preview}"
                    
                    response_text += "\n\n"
                
                return response_text.strip()
            else:
                return f"❌ Search error: {response.status_code}"
                
        except Exception as e:
            return f"❌ Search error: {str(e)}"
            
    async def handle_context_analysis(self, message: str) -> str:
        """Handle context analysis requests using AI."""
        try:
            response = await self.client.post(
                f"{MCP_SERVER_URL}/analyze_screen_context",
                json={
                    "query": message,
                    "max_events": 5
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result["events_analyzed"] == 0:
                    return "🤔 I couldn't find any relevant screen captures to analyze for your question."
                
                confidence_emoji = "🎯" if result["confidence"] > 0.7 else "💭"
                
                response_text = f"{confidence_emoji} {result['answer']}\n\n"
                response_text += f"📊 Analyzed {result['events_analyzed']} screen capture(s) "
                response_text += f"in {result.get('search_time', 0):.2f}s"
                
                return response_text
            else:
                return f"❌ Analysis error: {response.status_code}"
                
        except Exception as e:
            return f"❌ Analysis error: {str(e)}"
    
    def run(self):
        """Start the chat bot interface."""
        logger.info("Starting Screen Memory Chat Bot...")
        
        # Check if MCP server is running
        try:
            import requests
            response = requests.get(f"{MCP_SERVER_URL}/health", timeout=2)
            if response.status_code != 200:
                self.show_server_warning()
        except:
            self.show_server_warning()
        
        # Start the UI
        self.root.mainloop()
        
    def show_server_warning(self):
        """Show warning if MCP server is not running."""
        self.add_message("assistant", 
            "⚠️ Warning: I couldn't connect to the Screen Memory server.\n\n"
            "To start the server, run:\n"
            "python mcp_server.py\n\n"
            "Some features may not work until the server is running."
        )
        
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'client'):
            asyncio.run(self.client.aclose())


def create_chat_shortcut():
    """Create macOS shortcut for the chat bot."""
    shortcut_script = '''
#!/bin/zsh
cd /path/to/hack_mcp
source .venv/bin/activate
python chat_bot.py
'''
    
    # Save shortcut script
    script_path = os.path.expanduser("~/chat_shortcut.sh")
    with open(script_path, "w") as f:
        f.write(shortcut_script.replace("/path/to/hack_mcp", os.getcwd()))
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    print(f"Chat shortcut created at: {script_path}")
    print("To set up keyboard shortcut:")
    print("1. Open macOS Shortcuts app")
    print("2. Create new shortcut")
    print("3. Add 'Run Shell Script' action")
    print(f"4. Set script to: {script_path}")
    print("5. Assign keyboard shortcut (e.g., Cmd+Shift+C)")


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--create-shortcut":
        create_chat_shortcut()
        return
    
    try:
        # Create and run the chat bot
        chat_bot = ScreenMemoryChatBot()
        chat_bot.run()
    except KeyboardInterrupt:
        logger.info("Chat bot stopped by user")
    except Exception as e:
        logger.error(f"Chat bot error: {e}")
        messagebox.showerror("Error", f"Chat bot error: {e}")


if __name__ == "__main__":
    main() 