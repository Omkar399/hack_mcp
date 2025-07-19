#!/usr/bin/env python3
"""
Simple Interactive Screen Memory Chat
Capture, store, and search your screen memory!
"""
import asyncio
import json
import os
from datetime import datetime
from capture import ScreenCapture
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Simple in-memory storage
memory_db = []

def save_memory(capture_data):
    """Save capture to simple JSON file"""
    entry = {
        'timestamp': datetime.now().isoformat(),
        'window_title': capture_data.get('window_title', 'Unknown'),
        'app_name': str(capture_data.get('app_name', 'Unknown')),
        'text': capture_data.get('full_text', ''),
        'confidence': capture_data.get('ocr_conf', 0),
        'image_path': capture_data.get('image_path'),
    }
    memory_db.append(entry)
    
    # Also save to file
    with open('screen_memory.json', 'w') as f:
        json.dump(memory_db, f, indent=2)
    
    return entry

def search_memory(query):
    """Search memory for text"""
    results = []
    query_lower = query.lower()
    
    for entry in memory_db:
        if query_lower in entry['text'].lower():
            results.append(entry)
    
    return sorted(results, key=lambda x: x['timestamp'], reverse=True)

def load_existing_memory():
    """Load any existing memory file"""
    global memory_db
    try:
        if os.path.exists('screen_memory.json'):
            with open('screen_memory.json', 'r') as f:
                memory_db = json.load(f)
            console.print(f"📚 Loaded {len(memory_db)} existing memories")
    except:
        pass

async def main():
    console.print(Panel.fit("""
🖥️  Screen Memory Assistant - Interactive Demo

Commands:
• 'capture' or 'c' - Take a screenshot and remember it
• 'search <text>' or 's <text>' - Search your memory
• 'recent' or 'r' - Show recent captures
• 'stats' - Show memory statistics
• 'quit' or 'q' - Exit

Example: search "docker run" or search "error"
    """, title="Welcome!", border_style="green"))
    
    # Load existing memory
    load_existing_memory()
    
    # Initialize capture system
    capture_system = ScreenCapture()
    console.print("✅ Screen capture system ready!")
    
    while True:
        try:
            command = console.input("\n🤖 What would you like to do? ").strip()
            
            if command.lower() in ['quit', 'q', 'exit']:
                console.print("👋 Goodbye!")
                break
                
            elif command.lower() in ['capture', 'c']:
                console.print("📸 Capturing your screen...")
                capture_data = await capture_system.capture_screen(save_image=True)
                entry = save_memory(capture_data)
                
                console.print(Panel.fit(f"""
✅ Memory Saved!

🪟 Window: {entry['window_title']}
📱 App: {entry['app_name']}
📄 Text: {len(entry['text'])} characters
🔍 OCR Confidence: {entry['confidence']}%
📸 Image: {entry['image_path'] or 'Not saved'}
                """, title="Captured!", border_style="green"))
                
                # Show text preview
                if entry['text']:
                    preview = entry['text'][:200]
                    console.print(f"📝 Text Preview: {preview}...")
                
            elif command.lower().startswith('search ') or command.lower().startswith('s '):
                query = command.split(' ', 1)[1] if ' ' in command else ''
                if not query:
                    console.print("Please provide a search term!")
                    continue
                    
                results = search_memory(query)
                
                if not results:
                    console.print(f"🔍 No memories found for '{query}'")
                else:
                    table = Table(title=f"Search Results for '{query}'")
                    table.add_column("Time", width=20)
                    table.add_column("Window", width=25)
                    table.add_column("Match", width=50)
                    
                    for result in results[:5]:  # Show top 5
                        timestamp = result['timestamp'][:16].replace('T', ' ')
                        
                        # Find matching text snippet
                        text = result['text']
                        query_pos = text.lower().find(query.lower())
                        if query_pos >= 0:
                            start = max(0, query_pos - 50)
                            end = min(len(text), query_pos + len(query) + 50)
                            snippet = text[start:end]
                        else:
                            snippet = text[:100]
                            
                        table.add_row(
                            timestamp,
                            result['window_title'][:24],
                            f"...{snippet}..."
                        )
                    
                    console.print(table)
                
            elif command.lower() in ['recent', 'r']:
                if not memory_db:
                    console.print("📝 No memories yet - try 'capture' first!")
                else:
                    table = Table(title="Recent Memories")
                    table.add_column("Time", width=20)
                    table.add_column("Window", width=25) 
                    table.add_column("Text Preview", width=40)
                    table.add_column("OCR", width=8)
                    
                    for entry in memory_db[-5:]:  # Show last 5
                        timestamp = entry['timestamp'][:16].replace('T', ' ')
                        preview = entry['text'][:50] if entry['text'] else "(no text)"
                        
                        table.add_row(
                            timestamp,
                            entry['window_title'][:24],
                            f"{preview}...",
                            f"{entry['confidence']}%"
                        )
                    
                    console.print(table)
                    
            elif command.lower() == 'stats':
                if memory_db:
                    total_chars = sum(len(entry['text']) for entry in memory_db)
                    avg_confidence = sum(entry['confidence'] for entry in memory_db) / len(memory_db)
                    
                    console.print(Panel.fit(f"""
📊 Memory Statistics

Total Memories: {len(memory_db)}
Total Characters: {total_chars:,}
Average OCR Confidence: {avg_confidence:.1f}%
Storage File: screen_memory.json
                    """, title="Stats", border_style="blue"))
                else:
                    console.print("📝 No memories yet!")
                    
            else:
                console.print("❓ Unknown command. Type 'quit' to exit or try 'capture', 'search <text>', 'recent'")
                
        except KeyboardInterrupt:
            console.print("\n👋 Goodbye!")
            break
        except Exception as e:
            console.print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 