# MCP and Chat Integration Migration Guide

## Overview

This guide helps migrate Eidolon to version 0.2.0 with Model Context Protocol (MCP) server and enhanced chat functionality from the hack_mcp project.

## What's New in v0.2.0

### 1. MCP Server Integration
- Expose Eidolon's capabilities as MCP tools
- Compatible with Claude Desktop and other MCP clients
- Stdio transport for seamless integration

### 2. Enhanced Chat Interface
- Conversational access to screen memory
- Multi-model support (Gemini, Claude, OpenAI)
- Context-aware responses with screen capture data

### 3. Improved Capture System
- Hotkey support for manual captures
- Window tracking and app-specific capture
- Enhanced activity detection

## Installation Steps

### 1. Update Dependencies

```bash
# Update to latest requirements
pip install -r requirements.txt

# Or if using development mode
pip install -e .
```

### 2. Verify New Dependencies

```bash
# Check MCP framework
pip show enrichmcp

# Check enhanced capture tools  
pip show pyautogui pygetwindow keyboard
```

### 3. Update Configuration

Add the following to your `settings.yaml`:

```yaml
# MCP Server Configuration
mcp_server:
  enabled: true
  transport: "stdio"
  title: "eidolon-screen-memory"
  description: "Eidolon AI Personal Assistant with MCP support"
  max_events_context: 10
  search_limit: 5

# Chat Configuration  
chat:
  enabled: true
  default_model: "gemini-2.0-flash-exp"
  max_context_events: 10
  conversation_history_limit: 50
```

### 4. Update Environment Variables

Add to your `.env` file:

```env
# MCP Settings
MCP_ENABLED=true
MCP_TRANSPORT=stdio

# Chat Settings
CHAT_ENABLED=true
CHAT_DEFAULT_MODEL=gemini-2.0-flash-exp
```

## Testing the Integration

### 1. Test MCP Server

```bash
# Start MCP server
eidolon-mcp

# In another terminal, test with MCP client
# The server should respond to tool list requests
```

### 2. Test Chat Interface

```bash
# Start chat interface
eidolon-chat

# Try queries like:
# "What was I working on this morning?"
# "Show me my recent Python files"
# "Summarize my last hour of activity"
```

### 3. Test Enhanced Capture

```python
# Test programmatically
from eidolon import Observer
observer = Observer()

# Test hotkey capture (if configured)
# Press configured hotkey (default: Ctrl+Shift+S)
```

## Claude Desktop Integration

To use Eidolon with Claude Desktop:

1. Add to Claude Desktop config:
```json
{
  "servers": {
    "eidolon": {
      "command": "eidolon-mcp",
      "type": "stdio"
    }
  }
}
```

2. Restart Claude Desktop
3. Eidolon tools will be available in conversations

## Troubleshooting

### MCP Server Won't Start

```bash
# Check for port conflicts
lsof -i :8000

# Verify enrichmcp installation
python -c "import enrichmcp; print(enrichmcp.__version__)"

# Check logs
tail -f logs/eidolon.log
```

### Chat Models Not Working

```bash
# Verify API keys
python -c "import os; print('Keys set:', all([os.getenv(k) for k in ['GEMINI_API_KEY', 'CLAUDE_API_KEY']]))"

# Test model directly
eidolon chat --model gemini --test
```

### Import Errors

```bash
# Reinstall with all dependencies
pip install -e . --force-reinstall

# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
```

## Rollback Instructions

If you need to rollback to v0.1.0:

```bash
# Restore old requirements
git checkout v0.1.0 -- requirements.txt

# Reinstall
pip install -r requirements.txt

# Remove new config sections from settings.yaml
# Remove MCP and chat related entries
```

## Feature Comparison

| Feature | v0.1.0 | v0.2.0 |
|---------|---------|---------|
| Screenshot Capture | ✅ | ✅ Enhanced |
| OCR Processing | ✅ | ✅ |
| AI Analysis | ✅ | ✅ |
| Vector Search | ✅ | ✅ |
| MCP Server | ❌ | ✅ |
| Chat Interface | ❌ | ✅ |
| Hotkey Support | ❌ | ✅ |
| Window Tracking | ❌ | ✅ |
| Multi-model Chat | ❌ | ✅ |

## Next Steps

1. **Configure MCP tools**: Enable/disable specific tools based on your needs
2. **Set up chat models**: Add API keys for your preferred models
3. **Customize prompts**: Adjust system prompts for your use case
4. **Test with clients**: Try with Claude Desktop or other MCP clients
5. **Monitor performance**: Check resource usage with new features

## Support

For issues or questions:
- Check `logs/eidolon.log` for detailed error messages
- Review configuration in `settings.yaml`
- Ensure all API keys are properly set
- Verify Python version compatibility (3.9+)