#!/bin/bash
# Start MCP Chat System for Screen Memory Assistant

echo "🚀 Starting Screen Memory Assistant with MCP..."

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Please run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check if required files exist
if [ ! -f "mcp_server.py" ]; then
    echo "❌ mcp_server.py not found"
    exit 1
fi

if [ ! -f "chat_bot.py" ]; then
    echo "❌ chat_bot.py not found"  
    exit 1
fi

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✅ Environment variables loaded"
fi

echo ""
echo "📋 Instructions:"
echo "1. The chat bot will open"
echo "2. Click 'Connect to MCP Server'"
echo "3. Wait for connection confirmation"
echo "4. Start chatting with your screen memory!"
echo ""
echo "🔧 Available commands:"
echo "  • 'Search for Python code'"
echo "  • 'Take a screenshot'"
echo "  • 'Show recent captures'"
echo ""

# Start the chat bot (which will start the MCP server when connecting)
python chat_bot.py 