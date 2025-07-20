# Eidolon Configuration Guide

## Overview

Eidolon uses a hierarchical configuration system with environment variables taking precedence over configuration files. This guide covers all configuration options available in version 0.2.0 with MCP integration.

## Configuration Files

1. **`eidolon/config/settings.yaml`** - Main configuration file
2. **`.env`** - Environment variables (created from `.env.example`)
3. **`pyproject.toml`** - Project configuration and dependencies

## MCP Server Configuration

The Model Context Protocol (MCP) server enables Eidolon to act as a tool provider for AI assistants.

### Basic MCP Settings

```yaml
mcp_server:
  enabled: true                      # Enable/disable MCP server
  transport: "stdio"                 # Transport type (stdio, websocket, http)
  title: "eidolon-screen-memory"     # Server identification
  description: "Eidolon AI Personal Assistant with MCP support"
```

### MCP Tool Configuration

```yaml
mcp_server:
  tools:
    search_enabled: true             # Enable screen content search
    status_enabled: true             # Enable system status queries
    capture_enabled: true            # Enable manual capture trigger
    analyze_enabled: true            # Enable AI analysis requests
```

### MCP Performance Settings

```yaml
mcp_server:
  max_events_context: 10             # Max events to include in context
  search_limit: 5                    # Default search result limit
  batch_size: 10                     # Processing batch size
  timeout_seconds: 30                # Tool execution timeout
```

## Chat Configuration

The chat interface provides conversational access to Eidolon's capabilities.

### Basic Chat Settings

```yaml
chat:
  enabled: true                      # Enable chat functionality
  default_model: "gemini-2.0-flash-exp"  # Default AI model
  max_context_events: 10             # Events to include in context
  conversation_history_limit: 50     # Messages to retain
```

### Model-Specific Configuration

```yaml
chat:
  models:
    gemini:
      enabled: true
      model_name: "gemini-2.0-flash-exp"
      max_tokens: 8192
      temperature: 0.7
    claude:
      enabled: true
      model_name: "claude-3-opus-20240229"
      max_tokens: 4096
      temperature: 0.7
    openai:
      enabled: true
      model_name: "gpt-4-turbo-preview"
      max_tokens: 4096
      temperature: 0.7
```

### Chat Interface Settings

```yaml
chat:
  suggestions_enabled: true          # Show query suggestions
  context_preview_length: 200        # Context preview characters
  stream_responses: true             # Enable response streaming
  auto_save_conversations: true      # Save chat history
```

### System Prompt Configuration

```yaml
chat:
  system_prompt_template: |
    You are Eidolon, an AI personal assistant that helps users understand their screen activity.
    Use the provided screen capture context to answer questions about their work and activities.
    Be helpful, concise, and accurate in your responses.
```

## Environment Variables

Environment variables override configuration file settings. Create a `.env` file from `.env.example`:

### API Keys
```env
GEMINI_API_KEY=your_key_here
CLAUDE_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

### MCP Settings
```env
MCP_ENABLED=true
MCP_TRANSPORT=stdio
MCP_MAX_CONTEXT_EVENTS=10
```

### Chat Settings
```env
CHAT_ENABLED=true
CHAT_DEFAULT_MODEL=gemini-2.0-flash-exp
CHAT_STREAM_RESPONSES=true
CHAT_MAX_HISTORY=50
```

### Enhanced Capture Settings
```env
CAPTURE_HOTKEY=ctrl+shift+s
AUTO_CAPTURE_ON_ACTIVITY=true
WINDOW_TRACKING_ENABLED=true
```

## CLI Entry Points

With MCP integration, Eidolon provides multiple CLI commands:

- `eidolon` - Main CLI interface
- `eidolon-mcp` - Start MCP server
- `eidolon-chat` - Start chat interface

## Migration from v0.1.0

When upgrading from v0.1.0 to v0.2.0:

1. **Update dependencies**: Run `pip install -r requirements.txt` to install new dependencies
2. **Update configuration**: Add MCP and chat sections to `settings.yaml`
3. **Environment variables**: Update `.env` file with new variables
4. **Test integration**: Run `eidolon status` to verify configuration

## Troubleshooting

### MCP Server Issues

If the MCP server fails to start:
1. Check `enrichmcp` is installed: `pip show enrichmcp`
2. Verify transport setting matches your client
3. Check logs for binding errors

### Chat Model Issues

If chat models fail:
1. Verify API keys are set correctly
2. Check model names match provider's current offerings
3. Monitor API rate limits and quotas

### Performance Issues

If experiencing performance problems:
1. Reduce `max_events_context` and `batch_size`
2. Disable unused tools in MCP configuration
3. Use local models when possible

## Best Practices

1. **Security**: Keep API keys in `.env`, never commit them
2. **Performance**: Start with conservative settings, increase as needed
3. **Privacy**: Review and adjust sensitive patterns regularly
4. **Monitoring**: Enable logging and check regularly
5. **Updates**: Back up configuration before updates

## Configuration Schema Reference

For complete configuration schema, see:
- `eidolon/config/settings.yaml` - Fully commented configuration
- `docs/API.md` - API documentation
- `CLAUDE.md` - Technical specifications