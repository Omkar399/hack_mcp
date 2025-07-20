"""
Native macOS Chat Bot for Screen Memory Assistant

Uses proper MCP client to communicate with our EnrichMCP server.
Provides context-aware AI responses using captured screen data.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import asyncio
import json
import subprocess
import sys
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPClient:
    """Simple MCP client that communicates with our EnrichMCP server via subprocess"""
    
    def __init__(self, server_command: List[str]):
        self.server_command = server_command
        self.process = None
        self.request_id = 0
        
    async def start(self):
        """Start the MCP server subprocess"""
        try:
            self.process = await asyncio.create_subprocess_exec(
                *self.server_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            logger.info("MCP server started")
            
            # Initialize the connection
            await self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": True},
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "screen-memory-chat-bot",
                    "version": "1.0.0"
                }
            })
            
            return True
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            return False
    
    async def stop(self):
        """Stop the MCP server subprocess"""
        if self.process:
            self.process.terminate()
            await self.process.wait()
            self.process = None
    
    async def _send_request(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send an MCP request and get response"""
        if not self.process:
            raise RuntimeError("MCP server not started")
        
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params
        }
        
        # Send request
        request_str = json.dumps(request) + "\n"
        self.process.stdin.write(request_str.encode())
        await self.process.stdin.drain()
        
        # Read response
        response_line = await self.process.stdout.readline()
        if not response_line:
            raise RuntimeError("No response from MCP server")
        
        response = json.loads(response_line.decode())
        
        if "error" in response:
            raise RuntimeError(f"MCP error: {response['error']}")
        
        return response.get("result", {})
    
    async def list_resources(self) -> List[Dict[str, Any]]:
        """List available MCP resources"""
        try:
            result = await self._send_request("resources/list", {})
            return result.get("resources", [])
        except Exception as e:
            logger.error(f"Failed to list resources: {e}")
            return []
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available MCP tools"""
        try:
            result = await self._send_request("tools/list", {})
            return result.get("tools", [])
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            return []
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool"""
        try:
            result = await self._send_request("tools/call", {
                "name": name,
                "arguments": arguments
            })
            return result
        except Exception as e:
            logger.error(f"Failed to call tool {name}: {e}")
            return {"error": str(e)}


class ScreenMemoryChatBot:
    """Native macOS chat bot with MCP integration"""
    
    def __init__(self):
        self.window = None
        self.chat_display = None
        self.input_field = None
        self.send_button = None
        self.status_label = None
        
        # MCP client
        self.mcp_client = None
        self.mcp_connected = False
        
        # Chat history
        self.chat_history = []
        
        # OpenRouter API setup
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.openrouter_api_key:
            logger.warning("OPENROUTER_API_KEY not found. AI responses will be limited.")
    
    def create_ui(self):
        """Create the chat bot UI"""
        self.window = tk.Tk()
        self.window.title("Screen Memory Assistant - Chat Bot")
        self.window.geometry("600x700")
        
        # Configure styles
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Status.TLabel', font=('Arial', 10))
        
        # Title
        title_label = ttk.Label(
            self.window, 
            text="🧠 Screen Memory Assistant", 
            style='Title.TLabel'
        )
        title_label.pack(pady=10)
        
        # Status
        self.status_label = ttk.Label(
            self.window, 
            text="🔴 Disconnected from MCP server", 
            style='Status.TLabel'
        )
        self.status_label.pack(pady=5)
        
        # Connect button
        connect_button = ttk.Button(
            self.window,
            text="Connect to MCP Server",
            command=self.connect_mcp
        )
        connect_button.pack(pady=5)
        
        # Chat display
        chat_frame = ttk.Frame(self.window)
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
            height=20,
            font=('Arial', 11)
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        
        # Input frame
        input_frame = ttk.Frame(self.window)
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.input_field = ttk.Entry(input_frame, font=('Arial', 11))
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.input_field.bind('<Return>', lambda e: self.send_message())
        
        self.send_button = ttk.Button(
            input_frame,
            text="Send",
            command=self.send_message,
            state=tk.DISABLED
        )
        self.send_button.pack(side=tk.RIGHT)
        
        # Initial message
        self.add_message("System", "Welcome! Connect to the MCP server to start chatting with your screen memory.")
    
    def add_message(self, sender: str, message: str):
        """Add a message to the chat display"""
        self.chat_display.config(state=tk.NORMAL)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Format the message
        if sender == "System":
            self.chat_display.insert(tk.END, f"[{timestamp}] 🤖 {message}\n\n")
        elif sender == "User":
            self.chat_display.insert(tk.END, f"[{timestamp}] 👤 {message}\n\n")
        elif sender == "Assistant":
            self.chat_display.insert(tk.END, f"[{timestamp}] 🧠 {message}\n\n")
        else:
            self.chat_display.insert(tk.END, f"[{timestamp}] {sender}: {message}\n\n")
        
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def connect_mcp(self):
        """Connect to the MCP server"""
        def connect_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Create MCP client
                server_command = [sys.executable, "mcp_server.py"]
                self.mcp_client = MCPClient(server_command)
                
                # Start the server
                success = loop.run_until_complete(self.mcp_client.start())
                
                if success:
                    # Update UI in main thread
                    self.window.after(0, self._on_mcp_connected)
                    
                    # List available resources and tools
                    resources = loop.run_until_complete(self.mcp_client.list_resources())
                    tools = loop.run_until_complete(self.mcp_client.list_tools())
                    
                    # Show capabilities in chat
                    self.window.after(0, lambda: self._show_capabilities(resources, tools))
                else:
                    self.window.after(0, lambda: self.add_message("System", "❌ Failed to connect to MCP server"))
                    
            except Exception as e:
                logger.error(f"MCP connection failed: {e}")
                self.window.after(0, lambda: self.add_message("System", f"❌ Connection error: {e}"))
            finally:
                loop.close()
        
        # Run connection in background thread
        thread = threading.Thread(target=connect_async, daemon=True)
        thread.start()
    
    def _on_mcp_connected(self):
        """Called when MCP connection is established"""
        self.mcp_connected = True
        self.status_label.config(text="🟢 Connected to MCP server")
        self.send_button.config(state=tk.NORMAL)
        self.add_message("System", "✅ Connected to MCP server successfully!")
    
    def _show_capabilities(self, resources: List[Dict], tools: List[Dict]):
        """Show MCP server capabilities"""
        message = "🔧 Available capabilities:\n\n"
        
        if resources:
            message += "📄 Resources:\n"
            for resource in resources[:5]:  # Show first 5
                name = resource.get('name', 'Unknown')
                description = resource.get('description', 'No description')
                message += f"  • {name}: {description}\n"
            if len(resources) > 5:
                message += f"  ... and {len(resources) - 5} more\n"
            message += "\n"
        
        if tools:
            message += "🛠️ Tools:\n"
            for tool in tools[:5]:  # Show first 5
                name = tool.get('name', 'Unknown')
                description = tool.get('description', 'No description')
                message += f"  • {name}: {description}\n"
            if len(tools) > 5:
                message += f"  ... and {len(tools) - 5} more\n"
        
        message += "\nYou can now ask questions about your screen captures!"
        self.add_message("System", message)
    
    def send_message(self):
        """Send a message and get AI response"""
        if not self.mcp_connected:
            self.add_message("System", "❌ Please connect to MCP server first")
            return
        
        user_message = self.input_field.get().strip()
        if not user_message:
            return
        
        # Clear input
        self.input_field.delete(0, tk.END)
        
        # Add user message
        self.add_message("User", user_message)
        
        # Process message in background
        def process_message():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                response = loop.run_until_complete(self._process_user_message(user_message))
                self.window.after(0, lambda: self.add_message("Assistant", response))
            except Exception as e:
                logger.error(f"Message processing failed: {e}")
                self.window.after(0, lambda: self.add_message("System", f"❌ Error: {e}"))
            finally:
                loop.close()
        
        thread = threading.Thread(target=process_message, daemon=True)
        thread.start()
    
    async def _process_user_message(self, message: str) -> str:
        """Process user message and generate response using MCP tools"""
        try:
            # Simple keyword-based tool selection
            if any(word in message.lower() for word in ['search', 'find', 'look for']):
                # Extract search query (simple approach)
                query = message.lower()
                for word in ['search for', 'find', 'look for']:
                    if word in query:
                        query = query.split(word, 1)[1].strip()
                        break
                
                # Call search tool
                result = await self.mcp_client.call_tool("search_screens", {"query": query, "limit": 5})
                
                if "error" in result:
                    return f"Search failed: {result['error']}"
                
                # Format search results
                screens = result.get("content", [])
                if not screens:
                    return f"No screen captures found matching '{query}'"
                
                response = f"Found {len(screens)} screen captures matching '{query}':\n\n"
                for i, screen in enumerate(screens[:3], 1):
                    timestamp = screen.get("timestamp", "Unknown")
                    window_title = screen.get("window_title", "Unknown")
                    text_preview = screen.get("full_text", "")[:100] + "..." if screen.get("full_text") else "No text"
                    response += f"{i}. {timestamp} - {window_title}\n   {text_preview}\n\n"
                
                return response
            
            elif any(word in message.lower() for word in ['capture', 'screenshot', 'take']):
                # Call capture tool
                result = await self.mcp_client.call_tool("capture_screen", {})
                
                if "error" in result:
                    return f"Capture failed: {result['error']}"
                
                return "📸 Screen captured successfully! The new capture has been saved and processed."
            
            elif any(word in message.lower() for word in ['recent', 'latest', 'last']):
                # Get recent captures
                result = await self.mcp_client.call_tool("get_recent_screens", {"limit": 5})
                
                if "error" in result:
                    return f"Failed to get recent captures: {result['error']}"
                
                screens = result.get("content", [])
                if not screens:
                    return "No recent screen captures found."
                
                response = f"Your {len(screens)} most recent screen captures:\n\n"
                for i, screen in enumerate(screens, 1):
                    timestamp = screen.get("timestamp", "Unknown")
                    window_title = screen.get("window_title", "Unknown")
                    app_name = screen.get("app_name", "Unknown")
                    response += f"{i}. {timestamp} - {app_name} ({window_title})\n"
                
                return response
            
            else:
                # General query - provide helpful guidance
                return """I can help you with your screen memory! Here's what you can ask:

🔍 **Search**: "Search for Python code" or "Find email from John"
📸 **Capture**: "Take a screenshot" or "Capture current screen"  
📋 **Recent**: "Show recent captures" or "What did I do lately?"

Your screen captures include OCR text, window titles, and visual context. Ask me anything about what you've been working on!"""
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"Sorry, I encountered an error: {e}"
    
    def run(self):
        """Start the chat bot"""
        self.create_ui()
        
        # Handle window close
        def on_closing():
            if self.mcp_client:
                # Stop MCP client in background
                def stop_mcp():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.mcp_client.stop())
                    loop.close()
                
                thread = threading.Thread(target=stop_mcp, daemon=True)
                thread.start()
                time.sleep(0.5)  # Give it a moment to cleanup
            
            self.window.destroy()
        
        self.window.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Start the UI
        logger.info("Starting Screen Memory Chat Bot...")
        self.window.mainloop()


def main():
    """Main entry point"""
    chat_bot = ScreenMemoryChatBot()
    chat_bot.run()


if __name__ == "__main__":
    main() 