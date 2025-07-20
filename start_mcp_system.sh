#!/bin/bash

# Screen Memory Assistant - MCP System Startup Script
# This script starts the MCP server and provides chat bot access

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo -e "${BLUE}🚀 Screen Memory Assistant - MCP System${NC}"
echo "========================================"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}❌ Virtual environment not found. Please run 'uv sync' first.${NC}"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Check if database is running
echo -e "${YELLOW}🔍 Checking database...${NC}"
if ! docker ps | grep -q postgres; then
    echo -e "${YELLOW}🐳 Starting PostgreSQL database...${NC}"
    docker-compose up -d postgres
    sleep 3
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Creating template...${NC}"
    cat > .env << EOF
# OpenRouter API Key for AI features
OPENROUTER_API_KEY=your_api_key_here

# Database URL (default should work with Docker)
DATABASE_URL=postgresql+asyncpg://hack:hack123@localhost:5432/screenmemory

# Logging level
LOG_LEVEL=INFO
EOF
    echo -e "${YELLOW}📝 Please edit .env file with your OpenRouter API key${NC}"
fi

# Function to start MCP server
start_mcp_server() {
    echo -e "${GREEN}🖥️  Starting MCP Server...${NC}"
    echo "Server will be available at: http://localhost:8001"
    echo "Press Ctrl+C to stop the server"
    echo ""
    
    # Start the MCP server
    python mcp_server.py
}

# Function to start chat bot
start_chat_bot() {
    echo -e "${GREEN}💬 Starting Chat Bot...${NC}"
    python chat_bot.py
}

# Function to create shortcuts
create_shortcuts() {
    echo -e "${BLUE}🔧 Creating macOS shortcuts...${NC}"
    
    # Create chat shortcut
    python chat_bot.py --create-shortcut
    
    # Create capture shortcut (update existing one to work with MCP)
    cat > capture_mcp_shortcut.sh << EOF
#!/bin/zsh
cd "$PROJECT_DIR"
source .venv/bin/activate

# Check if MCP server is running
if ! curl -s http://localhost:8001/health > /dev/null 2>&1; then
    osascript -e 'display notification "MCP Server not running. Please start it first." with title "Screen Memory"'
    exit 1
fi

# Trigger capture via MCP server
curl -X POST http://localhost:8001/capture_screen \\
    -H "Content-Type: application/json" \\
    -d '{"save_image": true, "use_vision": false}' \\
    -s | python -c "
import json, sys
try:
    result = json.load(sys.stdin)
    if result.get('success'):
        print(f'Screen captured! Event ID: {result[\"event_id\"]}')
    else:
        print(f'Capture failed: {result.get(\"message\", \"Unknown error\")}')
except:
    print('Capture request failed')
"

# Show notification
osascript -e 'display notification "Screen captured and processed" with title "Screen Memory" sound name "Glass"'
EOF
    
    chmod +x capture_mcp_shortcut.sh
    
    echo -e "${GREEN}✅ Shortcuts created:${NC}"
    echo "  - Chat: ~/chat_shortcut.sh"
    echo "  - Capture: $PROJECT_DIR/capture_mcp_shortcut.sh"
    echo ""
    echo -e "${YELLOW}📱 To set up keyboard shortcuts:${NC}"
    echo "1. Open macOS Shortcuts app"
    echo "2. Create shortcuts for each script"
    echo "3. Assign keyboard shortcuts:"
    echo "   - Cmd+Shift+C for chat"
    echo "   - Cmd+Shift+S for capture"
}

# Function to show status
show_status() {
    echo -e "${BLUE}📊 System Status${NC}"
    echo "==============="
    
    # Check database
    if docker ps | grep -q postgres; then
        echo -e "${GREEN}✅ Database: Running${NC}"
    else
        echo -e "${RED}❌ Database: Not running${NC}"
    fi
    
    # Check MCP server
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ MCP Server: Running (http://localhost:8001)${NC}"
    else
        echo -e "${RED}❌ MCP Server: Not running${NC}"
    fi
    
    # Check environment
    if [ -f ".env" ] && grep -q "OPENROUTER_API_KEY=your_api_key_here" .env; then
        echo -e "${YELLOW}⚠️  OpenRouter API: Not configured${NC}"
    elif [ -f ".env" ]; then
        echo -e "${GREEN}✅ OpenRouter API: Configured${NC}"
    else
        echo -e "${RED}❌ Environment: .env file missing${NC}"
    fi
}

# Function to test the system
test_system() {
    echo -e "${BLUE}🧪 Testing System${NC}"
    echo "=================="
    
    # Test database connection
    echo -e "${YELLOW}Testing database connection...${NC}"
    if python -c "
import asyncio
from database import db

async def test():
    try:
        await db.initialize()
        health = await db.health_check()
        print('✅ Database connection: OK' if health else '❌ Database connection: Failed')
    except Exception as e:
        print(f'❌ Database error: {e}')

asyncio.run(test())
"; then
        echo -e "${GREEN}Database test completed${NC}"
    else
        echo -e "${RED}Database test failed${NC}"
    fi
    
    # Test capture system
    echo -e "${YELLOW}Testing capture system...${NC}"
    python -c "
import asyncio
from capture import ScreenCapture

async def test():
    try:
        capture = ScreenCapture()
        result = await capture.capture_screen(save_image=False)
        print('✅ Screen capture: OK')
        print(f'   OCR confidence: {result.get(\"ocr_conf\", 0)}%')
        print(f'   Text length: {len(result.get(\"full_text\", \"\"))} chars')
    except Exception as e:
        print(f'❌ Capture error: {e}')

asyncio.run(test())
"
}

# Main menu
case "${1:-}" in
    "server")
        start_mcp_server
        ;;
    "chat")
        start_chat_bot
        ;;
    "shortcuts")
        create_shortcuts
        ;;
    "status")
        show_status
        ;;
    "test")
        test_system
        ;;
    "stop")
        echo -e "${YELLOW}🛑 Stopping services...${NC}"
        pkill -f "python mcp_server.py" || true
        pkill -f "python chat_bot.py" || true
        echo -e "${GREEN}✅ Services stopped${NC}"
        ;;
    *)
        echo -e "${GREEN}Usage: $0 {server|chat|shortcuts|status|test|stop}${NC}"
        echo ""
        echo "Commands:"
        echo "  server     - Start the MCP server"
        echo "  chat       - Start the chat bot interface"
        echo "  shortcuts  - Create macOS keyboard shortcuts"
        echo "  status     - Show system status"
        echo "  test       - Test system components"
        echo "  stop       - Stop all services"
        echo ""
        echo "Quick start:"
        echo "1. $0 server     (in one terminal)"
        echo "2. $0 chat       (in another terminal)"
        echo "3. $0 shortcuts  (to set up keyboard shortcuts)"
        ;;
esac 