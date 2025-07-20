#!/usr/bin/env python3
"""
LLM-Powered Screen Memory Chat

Real AI chat that analyzes your screen captures using GPT-4 via OpenRouter
"""

import asyncio
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import httpx

# Load environment
load_dotenv()

# Database and capture imports
from database import db

class LLMScreenMemoryChat:
    """LLM-powered chat for screen memory analysis"""
    
    def __init__(self):
        # Try to get API key from environment first, then fallback to hardcoded
        self.api_key = os.getenv("GEMINI_API_KEY") or "AIzaSyAUlyDM3-dP2anUZhuOWguBMZTPYaAlz0Q"
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.model = "gemini-2.0-flash-exp"
        
        if not self.api_key:
            raise ValueError("No Gemini API key found!")
    
    async def initialize(self):
        """Initialize database and systems"""
        print("🚀 Initializing LLM Screen Memory Chat...")
        
        # Initialize database
        await db.initialize()
        print("✅ Database connected")
        
        # Test API connection
        await self.test_api()
        print("✅ LLM API ready")
        
        print("✅ System ready!")
    
    async def test_api(self):
        """Test Gemini API connection"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/models/{self.model}:generateContent",
                    params={"key": self.api_key},
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": [{"text": "Hi, just testing connection"}]}],
                        "generationConfig": {"maxOutputTokens": 10}
                    },
                    timeout=10.0
                )
                if response.status_code != 200:
                    error_msg = f"API test failed: {response.status_code}"
                    try:
                        error_data = response.json()
                        if 'error' in error_data:
                            error_msg += f" - {error_data['error'].get('message', 'Unknown error')}"
                    except:
                        error_msg += f" - {response.text[:200]}"
                    raise Exception(error_msg)
        except Exception as e:
            raise Exception(f"API connection failed: {e}")
    
    async def get_context_from_query(self, query: str) -> str:
        """Get relevant screen capture context for the query"""
        # Search recent captures
        recent_events = await db.get_recent_events(limit=10)
        
        # Search for query terms
        search_results = await db.search_events(query, limit=5)
        
        # Combine and format context
        context_parts = []
        
        if recent_events:
            context_parts.append("## Recent Screen Activity:")
            for event in recent_events[:5]:
                context_parts.append(f"- **{event.app_name}** ({event.window_title})")
                if event.full_text and len(event.full_text.strip()) > 10:
                    preview = event.full_text[:200] + "..." if len(event.full_text) > 200 else event.full_text
                    context_parts.append(f"  Text: {preview}")
                context_parts.append(f"  Time: {event.ts}")
                context_parts.append("")
        
        if search_results:
            context_parts.append("## Relevant Screen Captures:")
            for event in search_results:
                context_parts.append(f"- **{event.app_name}** ({event.window_title})")
                if event.full_text and len(event.full_text.strip()) > 10:
                    preview = event.full_text[:300] + "..." if len(event.full_text) > 300 else event.full_text
                    context_parts.append(f"  Text: {preview}")
                context_parts.append(f"  Time: {event.ts}")
                context_parts.append("")
        
        if not context_parts:
            context_parts = ["No screen capture data found."]
        
        return "\n".join(context_parts)
    
    async def call_llm(self, messages: list) -> str:
        """Call Gemini API"""
        try:
            # Convert OpenAI-style messages to Gemini format
            gemini_contents = []
            for msg in messages:
                if msg["role"] == "system":
                    # Prepend system message to the first user message
                    continue
                elif msg["role"] == "user":
                    # Include system context if available
                    system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
                    content = f"{system_msg}\n\nUser: {msg['content']}" if system_msg else msg["content"]
                    gemini_contents.append({"parts": [{"text": content}]})
                    break  # Only handle first user message for simplicity
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/models/{self.model}:generateContent",
                    params={"key": self.api_key},
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": gemini_contents,
                        "generationConfig": {
                            "maxOutputTokens": 1000,
                            "temperature": 0.7
                        }
                    },
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    error_text = response.text
                    raise Exception(f"API error {response.status_code}: {error_text}")
                
                result = response.json()
                return result["candidates"][0]["content"]["parts"][0]["text"]
                
        except Exception as e:
            return f"❌ LLM Error: {e}"
    
    async def process_query(self, user_query: str) -> str:
        """Process user query with LLM and screen context"""
        
        # Get relevant screen context
        context = await self.get_context_from_query(user_query)
        
        # Build system prompt
        system_prompt = f"""You are a Screen Memory Assistant. You help users understand and analyze their screen activity.

You have access to screen captures with OCR text from the user's computer. Use this information to answer their questions about what they were doing, what they saw, or to help them find specific information.

Current screen capture context:
{context}

Instructions:
- Be helpful and conversational
- Reference specific screen captures when relevant
- If no relevant data is found, suggest taking a screenshot
- Keep responses concise but informative
- Use emojis sparingly but appropriately
"""
        
        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        # Call LLM
        return await self.call_llm(messages)
    
    async def chat_loop(self):
        """Main chat loop"""
        await self.initialize()
        
        print("""
🧠 LLM-Powered Screen Memory Assistant

Ask me anything about your screen activity!
Examples:
- "What was I working on this morning?"
- "Find any Python code I was looking at"
- "What websites did I visit?"
- "Show me recent terminal commands"

Type 'quit' to exit.
""")
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                    
                if not user_input:
                    continue
                    
                print("🤔 Thinking...")
                response = await self.process_query(user_input)
                print(f"\n🧠 Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

async def main():
    """Main entry point"""
    try:
        chat = LLMScreenMemoryChat()
        await chat.chat_loop()
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        print("Make sure you have GEMINI_API_KEY set in .env file (or it's hardcoded)")

if __name__ == "__main__":
    asyncio.run(main()) 