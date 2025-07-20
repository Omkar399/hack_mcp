#!/usr/bin/env python3
"""
Screen Memory Assistant Demo

Demonstrates the EnrichMCP integration and chat bot capabilities.
"""

import os
import sys
import asyncio
import json
import time
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from capture import ScreenCapture
from database import db
from dotenv import load_dotenv

load_dotenv()


async def demo_capture_and_search():
    """Demonstrate screen capture and search capabilities."""
    print("🎬 Screen Memory Assistant Demo")
    print("=" * 40)
    
    # Initialize systems
    print("\n1. Initializing systems...")
    await db.initialize()
    capture_system = ScreenCapture()
    
    print("✅ Database initialized")
    print("✅ Capture system ready")
    
    # Capture current screen
    print("\n2. Capturing current screen...")
    start_time = time.time()
    
    result = await capture_system.capture_screen(save_image=True)
    event_id = await db.save_screen_event(result)
    
    capture_time = time.time() - start_time
    
    print(f"✅ Screen captured in {capture_time:.2f}s")
    print(f"   Event ID: {event_id}")
    print(f"   OCR Confidence: {result.get('ocr_conf', 0)}%")
    print(f"   Text Length: {len(result.get('full_text', ''))} characters")
    print(f"   Window: {result.get('window_title', 'Unknown')}")
    print(f"   App: {result.get('app_name', 'Unknown')}")
    
    # Show some extracted text
    text = result.get('full_text', '')
    if text:
        preview = text[:200] + "..." if len(text) > 200 else text
        print(f"   Text Preview: {preview}")
    
    # Search for recent events
    print("\n3. Searching recent events...")
    recent_events = await db.get_recent_events(limit=5, hours=1)
    
    print(f"✅ Found {len(recent_events)} recent events:")
    for i, event in enumerate(recent_events, 1):
        print(f"   {i}. [{event.ts.strftime('%H:%M:%S')}] {event.app_name or 'Unknown'}")
        if event.window_title:
            print(f"      Window: {event.window_title}")
        if event.full_text:
            text_preview = event.full_text[:50] + "..." if len(event.full_text) > 50 else event.full_text
            print(f"      Text: {text_preview}")
    
    # Demonstrate search
    print("\n4. Testing search functionality...")
    search_query = "screen"  # Search for screens containing "screen"
    search_results = await db.search_events(query=search_query, limit=3)
    
    print(f"✅ Search for '{search_query}' found {len(search_results)} results:")
    for i, event in enumerate(search_results, 1):
        print(f"   {i}. [{event.ts.strftime('%H:%M:%S')}] {event.app_name or 'Unknown'}")
        if event.full_text and search_query.lower() in event.full_text.lower():
            # Find the context around the search term
            text = event.full_text.lower()
            pos = text.find(search_query.lower())
            if pos >= 0:
                start = max(0, pos - 30)
                end = min(len(text), pos + 50)
                context = event.full_text[start:end]
                print(f"      Context: ...{context}...")


async def demo_mcp_features():
    """Demonstrate MCP server features."""
    print("\n" + "=" * 40)
    print("🔌 MCP Server Features Demo")
    print("=" * 40)
    
    # Import MCP models
    from mcp_server import ScreenEvent, CaptureResult, SearchResult
    
    print("\n✅ EnrichMCP Models Available:")
    print("   - ScreenEvent: Screen capture with OCR and metadata")
    print("   - CaptureResult: Capture operation results")
    print("   - SearchResult: Search results with events")
    print("   - Command: Extracted commands (future)")
    print("   - ErrorEvent: Extracted errors (future)")
    
    print("\n✅ MCP Resource Endpoints:")
    print("   - capture_screen: Trigger screen capture")
    print("   - search_screens: Search by text content")
    print("   - get_recent_screens: Get recent captures")
    print("   - get_screen_by_id: Get specific screen")
    print("   - find_screens_with_errors: Find error screens")
    print("   - analyze_screen_context: AI-powered analysis")
    
    print("\n✅ Advanced Features:")
    print("   - Server-side LLM sampling via EnrichMCP")
    print("   - Type-safe models with validation")
    print("   - Relationship navigation")
    print("   - Pagination support")
    print("   - Context-aware responses")


def demo_chat_bot():
    """Demonstrate chat bot capabilities."""
    print("\n" + "=" * 40)
    print("💬 Chat Bot Demo")
    print("=" * 40)
    
    print("\n✅ Native macOS Chat Interface:")
    print("   - Tkinter-based popup window")
    print("   - Always-on-top for easy access")
    print("   - Real-time conversation")
    print("   - Status indicators")
    
    print("\n✅ Intelligent Query Processing:")
    print("   - Capture requests: 'Capture my screen now'")
    print("   - Search requests: 'Find login forms from today'")
    print("   - Context analysis: 'What was I working on?'")
    print("   - Natural language understanding")
    
    print("\n✅ Example Conversations:")
    
    examples = [
        {
            "user": "Capture my screen with AI vision",
            "assistant": "✅ Screen captured successfully!\nEvent ID: 42\nProcessing time: 1.23s"
        },
        {
            "user": "What was I working on 10 minutes ago?",
            "assistant": "🎯 Based on your recent screen captures, you were working on implementing the EnrichMCP server integration. I found 3 relevant screens showing code in VS Code with database models and API endpoints.\n\n📊 Analyzed 3 screen capture(s) in 0.45s"
        },
        {
            "user": "Find error messages from today",
            "assistant": "🔍 Found 2 relevant screen capture(s):\n\n1. [14:23] Terminal - bash\n   Text: Error: Database connection failed...\n\n2. [15:45] VS Code - Python\n   Text: ModuleNotFoundError: No module named..."
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n   Example {i}:")
        print(f"   User: {example['user']}")
        print(f"   Assistant: {example['assistant']}")


def demo_shortcuts():
    """Demonstrate keyboard shortcuts."""
    print("\n" + "=" * 40)
    print("⌨️  Keyboard Shortcuts Demo")
    print("=" * 40)
    
    print("\n✅ macOS Integration:")
    print("   - Cmd+Shift+S: Instant screen capture")
    print("   - Cmd+Shift+C: Open chat bot")
    print("   - Native notifications with sound")
    print("   - Accessibility permissions handled")
    
    print("\n✅ Capture Workflow:")
    print("   1. Press Cmd+Shift+S")
    print("   2. Screen captured instantly")
    print("   3. OCR and analysis in background")
    print("   4. Notification shows success")
    print("   5. Data available for chat queries")
    
    print("\n✅ Chat Workflow:")
    print("   1. Press Cmd+Shift+C")
    print("   2. Chat window appears on top")
    print("   3. Ask questions about screens")
    print("   4. Get intelligent responses")
    print("   5. Context from all captures")


async def main():
    """Run the complete demo."""
    print("🚀 Welcome to Screen Memory Assistant!")
    print("This demo shows the complete EnrichMCP integration and chat bot system.\n")
    
    # Check if OpenRouter API key is configured
    if not os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY") == "your_api_key_here":
        print("⚠️  Note: OpenRouter API key not configured.")
        print("   AI analysis features will be limited.")
        print("   Set OPENROUTER_API_KEY in .env for full functionality.\n")
    
    try:
        # Run the demos
        await demo_capture_and_search()
        await demo_mcp_features()
        demo_chat_bot()
        demo_shortcuts()
        
        print("\n" + "=" * 40)
        print("🎉 Demo Complete!")
        print("=" * 40)
        
        print("\n🚀 To start using the system:")
        print("   1. ./start_mcp_system.sh server    (start MCP server)")
        print("   2. ./start_mcp_system.sh chat      (start chat bot)")
        print("   3. ./start_mcp_system.sh shortcuts (set up shortcuts)")
        
        print("\n📚 Key Features Implemented:")
        print("   ✅ EnrichMCP server with semantic models")
        print("   ✅ Screen capture with OCR and vision fallback")
        print("   ✅ PostgreSQL database with vector search")
        print("   ✅ Native macOS chat interface")
        print("   ✅ Keyboard shortcuts and notifications")
        print("   ✅ Context-aware AI responses")
        print("   ✅ Multi-modal search (text + visual)")
        print("   ✅ Real-time capture processing")
        
        print("\n🔮 Future Enhancements:")
        print("   - Command extraction from terminal screens")
        print("   - Error detection and classification")
        print("   - Calendar event extraction")
        print("   - Advanced visual similarity search")
        print("   - Multi-device synchronization")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Make sure the database is running: docker-compose up -d")


if __name__ == "__main__":
    asyncio.run(main()) 