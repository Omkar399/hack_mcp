# Screen Memory Assistant - MCP Implementation

## Overview

This system implements a **proper Model Context Protocol (MCP)** server and client for screen memory functionality. The implementation follows MCP standards and best practices.

## Architecture

### ✅ Correct MCP Implementation

- **MCP Server**: Uses EnrichMCP framework with stdio transport (standard for local development)
- **MCP Client**: Communicates via proper MCP protocol, not HTTP REST
- **Chat Bot**: Native macOS interface that uses MCP client to access screen data
- **Protocol**: Standard MCP JSON-RPC over stdio

### ❌ Previous Issues (Now Fixed)

- ~~Was using HTTP transport with FastAPI endpoints~~
- ~~Chat bot was making REST API calls~~
- ~~Not following MCP protocol standards~~

## Key Components

### 1. MCP Server (`mcp_server.py`)
```python
# Uses EnrichMCP framework correctly
app = EnrichMCP(
    title="screen-memory-assistant",
    description="AI-powered screen memory system"
)

# Runs with stdio transport (standard for MCP)
app.run()  # Uses stdio, not HTTP
```

**Features:**
- **Resources**: Access to screen capture data
- **Tools**: Screen capture, search, analysis functions  
- **Models**: Structured data models for screen events
- **Stdio Transport**: Standard MCP communication method

### 2. MCP Client (`chat_bot.py`)
```python
# Proper MCP client implementation
class MCPClient:
    async def _send_request(self, method: str, params: Dict) -> Dict:
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params
        }
        # Sends via subprocess stdio, not HTTP
```

**Features:**
- **JSON-RPC Protocol**: Proper MCP communication
- **Subprocess Management**: Starts/stops MCP server
- **Async Communication**: Non-blocking MCP calls
- **Native UI**: macOS-optimized interface

## Usage

### Quick Start
```bash
# Start the complete system
./start_mcp_chat.sh
```

This will:
1. Activate the virtual environment
2. Load environment variables
3. Open the chat bot interface
4. Allow you to connect to the MCP server

### Manual Usage

#### 1. Start MCP Server (Command Line)
```bash
source .venv/bin/activate
python mcp_server.py
```

#### 2. Start Chat Bot
```bash
python chat_bot.py
```

#### 3. Connect in Chat Bot
1. Click "Connect to MCP Server"
2. Wait for connection confirmation
3. Start asking questions!

## MCP Protocol Details

### Standard MCP Communication
```json
// Request
{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {}
}

// Response  
{
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "tools": [...]
    }
}
```

### Available MCP Methods
- `tools/list` - List available tools
- `tools/call` - Execute a tool
- `resources/list` - List available resources
- `resources/read` - Read resource content

### Available Tools
- `capture_screen` - Take a new screenshot
- `search_screens` - Search through captured screens
- `get_recent_screens` - Get recent captures
- `analyze_screen_context` - AI analysis of screen content

## Examples

### Search for Content
```
User: "Search for Python code"
→ Calls: tools/call { name: "search_screens", arguments: { query: "python" }}
```

### Take Screenshot  
```
User: "Take a screenshot"
→ Calls: tools/call { name: "capture_screen", arguments: {} }
```

### Get Recent Activity
```
User: "Show recent captures"  
→ Calls: tools/call { name: "get_recent_screens", arguments: { limit: 5 }}
```

## Benefits of Proper MCP Implementation

### ✅ Standards Compliance
- **Interoperable**: Works with any MCP-compatible client
- **Extensible**: Easy to add new tools and resources
- **Maintainable**: Clear separation between protocol and business logic

### ✅ Performance
- **Efficient**: Direct stdio communication
- **Lightweight**: No HTTP overhead
- **Responsive**: Async processing throughout

### ✅ Security
- **Local Only**: No network exposure by default
- **Process Isolation**: Server runs in separate process
- **Clean Shutdown**: Proper resource cleanup

### ✅ Developer Experience
- **Type Safety**: Pydantic models throughout
- **Error Handling**: Comprehensive error responses
- **Logging**: Detailed operation logging
- **Testing**: Easy to test individual components

## Integration with AI Clients

This MCP server can be used with:

- **Claude Desktop**: Via MCP configuration
- **OpenAI Agents**: Via MCP client libraries
- **Custom Applications**: Via MCP protocol
- **Our Chat Bot**: Native implementation included

### Claude Desktop Integration
```json
{
  "mcpServers": {
    "screen-memory": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "/path/to/hack_mcp"
    }
  }
}
```

## Environment Setup

### Required Environment Variables
```bash
# .env file
OPENROUTER_API_KEY=your_openrouter_key_here
DATABASE_URL=postgresql://user:pass@localhost/screendb
```

### Dependencies
- Python 3.11+
- EnrichMCP framework
- PostgreSQL (for persistence)
- macOS (for native features)

## Troubleshooting

### Connection Issues
```
❌ "Failed to connect to MCP server"
→ Check if mcp_server.py starts without errors
→ Verify virtual environment is activated
→ Check database connectivity
```

### Tool Execution Errors
```
❌ "Tool call failed" 
→ Check server logs for specific error
→ Verify required permissions (screen capture)
→ Ensure database is accessible
```

### Performance Issues
```
❌ "Slow responses"
→ Check OCR processing load
→ Verify database indexes
→ Consider limiting search results
```

## Next Steps

1. **Test the Implementation**: Run `./start_mcp_chat.sh` and try the examples
2. **Explore Capabilities**: Use the chat bot to discover available tools
3. **Integrate with Other Clients**: Try connecting from Claude or custom apps
4. **Extend Functionality**: Add new tools and resources as needed

The system now follows proper MCP standards and will work seamlessly with any MCP-compatible client! 🎉 