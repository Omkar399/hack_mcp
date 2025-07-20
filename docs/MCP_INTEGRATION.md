# Eidolon MCP Integration Documentation

## Overview

Eidolon now includes a complete Model Context Protocol (MCP) server integration using the EnrichMCP framework. This provides external applications with programmatic access to Eidolon's screen memory and analysis capabilities.

## Features Migrated from hack_mcp

### Core Functionality
- **Screen Capture**: Capture screenshots with OCR and AI analysis
- **Smart Search**: Full-text search across captured content with filters
- **Error Detection**: Automatic detection and classification of error messages
- **Context Analysis**: AI-powered analysis of user activities and workflows
- **Real-time Monitoring**: Start/stop screenshot monitoring remotely

### Enhanced Capabilities
- **Integrated AI**: Uses Eidolon's existing CloudAPIManager for better AI integration
- **Advanced Analysis**: Leverages Eidolon's enhanced Analyzer with error detection and command extraction
- **Unified Database**: Uses Eidolon's MetadataDatabase for consistent data storage
- **Configuration Management**: Integrated with Eidolon's configuration system

## Installation

### Prerequisites
1. Eidolon must be installed and configured
2. Install the EnrichMCP dependency:
```bash
pip install enrichmcp>=0.4.5
```

### Configuration
The MCP server is configured in `eidolon/config/settings.yaml`:

```yaml
mcp:
  enabled: true
  transport: "stdio"
  enable_screen_capture: true
  enable_search: true
  enable_error_detection: true
  enable_context_analysis: true
  enable_cloud_ai: true
  max_search_results: 50
  preferred_provider: "claude"
```

## Usage

### Starting the MCP Server

#### Via CLI
```bash
# Start with default settings (stdio transport)
python -m eidolon mcp

# Start with HTTP transport
python -m eidolon mcp --transport http --port 8080

# Start with auto-monitoring
python -m eidolon mcp --auto-monitor
```

#### Programmatically
```python
from eidolon.core.mcp_server import main as mcp_main
mcp_main()
```

### MCP Client Integration

The MCP server provides the following endpoints:

#### Screen Capture
```python
# Capture current screen
result = await client.retrieve("capture_screen", {
    "save_image": True,
    "analyze_content": True,
    "use_cloud_ai": False
})
```

#### Search Screens
```python
# Search captured content
results = await client.retrieve("search_screens", {
    "query": "git commit",
    "limit": 10,
    "since_hours": 24,
    "content_type": "terminal"
})
```

#### Context Analysis
```python
# Analyze screen context with AI
analysis = await client.retrieve("analyze_screen_context", {
    "query": "What was I working on 2 hours ago?",
    "max_events": 5,
    "use_cloud_ai": True
})
```

#### Error Detection
```python
# Find screens with errors
errors = await client.retrieve("find_screens_with_errors", {
    "severity": "high",
    "since_hours": 6,
    "limit": 10
})
```

#### System Control
```python
# Start monitoring
await client.retrieve("start_monitoring", {
    "capture_interval": 10
})

# Stop monitoring
await client.retrieve("stop_monitoring")

# Get system status
status = await client.retrieve("get_system_status")
```

## API Reference

### Data Models

#### ScreenEvent
Represents a captured screen with analysis:
```python
{
    "id": 123,
    "timestamp": "2024-01-15T10:30:00",
    "file_path": "/path/to/screenshot.png",
    "hash": "abc123...",
    "full_text": "Extracted OCR text...",
    "ocr_confidence": 0.95,
    "content_type": "terminal",
    "description": "Command line interface",
    "tags": ["terminal", "git", "python"]
}
```

#### CaptureResult
Result of screen capture operation:
```python
{
    "screenshot_id": 123,
    "success": True,
    "message": "Screen captured successfully",
    "processing_time": 1.23,
    "extracted_text": "OCR text...",
    "content_type": "terminal"
}
```

#### SearchResult
Search operation results:
```python
{
    "events": [ScreenEvent, ...],
    "total_results": 15,
    "query": "search query",
    "search_time": 0.45
}
```

#### AnalysisResult
AI context analysis results:
```python
{
    "answer": "Based on your screens, you were working on...",
    "confidence": 0.85,
    "events_analyzed": 3,
    "query": "What was I working on?",
    "provider": "claude"
}
```

#### ErrorEvent
Detected error information:
```python
{
    "id": 456,
    "screen_event_id": 123,
    "timestamp": "2024-01-15T10:30:00",
    "error_type": "exception",
    "error_message": "Python traceback detected",
    "severity": "high"
}
```

### Enhanced Features

#### Command Extraction
Automatically extracts command-line commands from terminal screens:
```python
# Commands are extracted from terminal content
commands = analyzer.extract_commands(text)
# Returns: [{"command": "git", "arguments": "commit -m 'fix'", ...}]
```

#### Error Detection
Comprehensive error pattern detection:
```python
# Errors are automatically detected and classified
errors = analyzer.detect_errors(text)
# Returns severity levels: critical, high, medium, low
```

#### Context Analysis
Deep analysis of user activities:
```python
# Analyze what the user was doing
context = analyzer.analyze_context(text, content_type)
# Returns activity type, insights, and detailed analysis
```

## Architecture

### Integration Points

1. **Observer**: Uses Eidolon's Observer for screen capture
2. **Analyzer**: Enhanced with error detection and command extraction
3. **Database**: Uses MetadataDatabase for persistent storage
4. **Cloud AI**: Integrates with CloudAPIManager for AI analysis
5. **Configuration**: Unified configuration management

### Data Flow

```
MCP Client Request
    ↓
EnrichMCP Framework
    ↓
Eidolon MCP Server
    ↓
Core Components (Observer, Analyzer, Database)
    ↓
Response with Analysis
```

### Enhanced Analyzer Capabilities

The migrated functionality includes new methods in the Analyzer class:

1. **detect_errors()**: Detects and classifies errors by severity
2. **extract_commands()**: Extracts command-line commands with metadata
3. **analyze_context()**: Provides deep activity analysis

## Security Considerations

### Data Privacy
- All processing happens locally by default
- Cloud AI usage is optional and configurable
- Sensitive patterns are automatically redacted
- Data retention policies are enforced

### Access Control
- MCP server runs with same permissions as Eidolon
- No authentication required for local stdio transport
- HTTP transport should be secured if exposed

### Configuration Security
- API keys stored in environment variables
- Sensitive data redaction patterns configurable
- Cloud AI usage can be completely disabled

## Troubleshooting

### Common Issues

#### EnrichMCP Import Error
```bash
pip install enrichmcp>=0.4.5
```

#### Database Connection Issues
Check database permissions and path:
```python
from eidolon.storage.metadata_db import MetadataDatabase
db = MetadataDatabase()
stats = db.get_statistics()  # Should not raise exception
```

#### Cloud AI Not Available
Verify API keys in environment:
```bash
export CLAUDE_API_KEY="your_key"
export OPENAI_API_KEY="your_key"
export GEMINI_API_KEY="your_key"
```

#### Transport Issues
For HTTP transport, ensure port is available:
```bash
netstat -an | grep 8080
```

### Debugging

Enable debug logging:
```bash
python -m eidolon mcp --verbose
```

Check MCP server status:
```python
status = await client.retrieve("get_system_status")
print(status)
```

## Performance Considerations

### Resource Usage
- Memory: ~500MB baseline, up to 2GB with AI models
- CPU: <5% average, up to 20% during analysis
- Storage: ~1GB per week of screenshots

### Optimization
- Local AI models are cached for faster inference
- Database queries are optimized with indexes
- Screenshot deduplication reduces storage
- Configurable analysis timeouts prevent hanging

## Migration Benefits

### From hack_mcp to Eidolon MCP
1. **Better Integration**: Uses existing Eidolon infrastructure
2. **Enhanced Analysis**: More sophisticated error detection and context analysis
3. **Unified Configuration**: Single configuration file for all settings
4. **Improved Performance**: Optimized database and AI usage
5. **Better Error Handling**: Comprehensive error detection and reporting
6. **Extensibility**: Built on Eidolon's modular architecture

### Backward Compatibility
The MCP API maintains compatibility with existing hack_mcp clients while providing enhanced functionality.

## Future Enhancements

### Planned Features
1. **Advanced Relationships**: Command-to-error correlation
2. **Timeline Reconstruction**: Project workflow analysis
3. **Productivity Insights**: Activity pattern analysis
4. **Team Collaboration**: Shared screen memory
5. **Plugin System**: Custom analysis plugins

### Extension Points
- Custom error pattern definitions
- Additional AI model integrations
- Custom analysis workflows
- External tool integrations

## Support

For issues related to MCP integration:
1. Check this documentation
2. Review configuration settings
3. Check logs for error messages
4. Verify all dependencies are installed
5. Test basic Eidolon functionality first

The MCP integration builds on Eidolon's solid foundation to provide powerful screen memory capabilities through a standardized protocol interface.