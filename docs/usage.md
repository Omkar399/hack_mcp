# Usage Guide

This guide covers how to use Eidolon AI Personal Assistant effectively.

## Getting Started

### Starting the System

```bash
# Start all components
python -m eidolon start

# Start in background mode
python -m eidolon start --background

# Using the convenience script
./scripts/start.sh
```

### Basic Commands

```bash
# Check system status
python -m eidolon status

# Search your digital history
python -m eidolon search "meeting notes"

# Interactive chat
python -m eidolon chat

# Stop the system
python -m eidolon stop

# Clean up old data
python -m eidolon cleanup --days 30
```

## Search Functionality

### Text Search

Search through OCR-extracted text from your screenshots:

```bash
# Search for specific terms
python -m eidolon search "Python code"
python -m eidolon search "email from John"
python -m eidolon search "terminal commands"

# Time-based searches
python -m eidolon search "error message" --since "2 hours ago"
python -m eidolon search "meeting" --date "2024-01-15"
```

### Semantic Search

Find content based on meaning, not just exact matches:

```bash
# Conceptual searches
python -m eidolon search "programming tutorials"
python -m eidolon search "financial documents"
python -m eidolon search "design mockups"
```

### Search Options

```bash
# Limit results
python -m eidolon search "code" --limit 10

# Search specific time ranges
python -m eidolon search "meeting" --since "yesterday"
python -m eidolon search "email" --between "2024-01-01" "2024-01-31"

# Include context
python -m eidolon search "bug fix" --context 5
```

## Interactive Chat

### Starting a Chat Session

```bash
python -m eidolon chat
```

### Example Conversations

**Finding Recent Activity:**
```
You: What was I working on this morning?
Eidolon: Based on your screen activity, you were primarily working on Python code, specifically debugging a Flask application. I can see you had VS Code open with several Python files and were running terminal commands for testing.
```

**Code Assistance:**
```
You: Show me any Python functions I wrote today
Eidolon: I found several Python functions in your recent activity:
1. `process_screenshot()` - for image processing
2. `extract_text()` - for OCR functionality
3. `search_memory()` - for semantic search
```

**Activity Summary:**
```
You: Summarize my day
Eidolon: Today you spent most of your time on software development. Key activities included:
- 3 hours coding in Python (Flask app development)
- 1 hour in meetings (Zoom calls)
- 30 minutes reading documentation
- Multiple terminal sessions for testing and debugging
```

### Chat Commands

Within the chat interface:

- `/search <query>` - Quick search
- `/status` - System status
- `/help` - Show available commands
- `/clear` - Clear conversation history
- `/quit` - Exit chat

## System Monitoring

### Status Information

```bash
python -m eidolon status
```

Shows:
- System uptime
- Screenshot capture status
- Memory usage
- Storage statistics
- Component health

### Health Checks

```bash
# Comprehensive health check
./scripts/health_check.sh

# Quick component check
python -c "from eidolon.core import Observer; print('Observer:', Observer().is_monitoring())"
```

## Data Management

### Storage Information

```bash
# View storage statistics
python -m eidolon status --storage

# Check database size
ls -lh data/database/
```

### Cleanup Operations

```bash
# Remove data older than 30 days
python -m eidolon cleanup --days 30

# Remove data older than 1 week
python -m eidolon cleanup --days 7

# Clean up specific data types
python -m eidolon cleanup --screenshots-only --days 14
python -m eidolon cleanup --ocr-only --days 7
```

### Backup and Export

```bash
# Backup all data
cp -r data/ backup_$(date +%Y%m%d)/

# Export search results
python -m eidolon search "important" --export results.json
```

## Configuration

### Runtime Configuration

```bash
# Change capture interval (seconds)
python -m eidolon config set capture_interval 60

# Change activity threshold
python -m eidolon config set activity_threshold 0.2

# View current configuration
python -m eidolon config show
```

### Environment Variables

Edit `.env` file for persistent configuration:

```bash
# Screenshot capture
CAPTURE_INTERVAL=30
ACTIVITY_THRESHOLD=0.1

# Storage limits
MAX_STORAGE_GB=10
DATA_RETENTION_DAYS=90

# Performance
MAX_CPU_PERCENT=80
MAX_MEMORY_MB=2048

# Privacy
AUTO_REDACT_SENSITIVE=true
REDACT_PATTERNS=password,ssn,credit_card
```

## Advanced Features

### MCP Server Integration

```bash
# Start MCP server
python -m eidolon mcp --transport stdio

# Start with HTTP transport
python -m eidolon mcp --transport http --port 8080
```

### API Integration

If you have cloud AI API keys configured:

```bash
# Use specific AI model for analysis
python -m eidolon search "code review" --model gemini
python -m eidolon chat --model claude
```

### Batch Operations

```bash
# Process multiple queries
echo -e "meeting notes\nPython code\nemail drafts" | python -m eidolon search --batch

# Export multiple searches
python -m eidolon search "work" --export work_results.json
python -m eidolon search "personal" --export personal_results.json
```

## Privacy and Security

### Data Redaction

Eidolon automatically redacts sensitive information:

- Credit card numbers
- Social security numbers
- Email addresses (optional)
- Phone numbers (optional)
- Custom patterns

### Local Processing

By default, all processing happens locally:

```bash
# Verify local-only mode
python -m eidolon config show | grep -i cloud
# Should show: cloud_ai_enabled=false
```

### Data Encryption

Screenshots and sensitive data are automatically encrypted at rest.

## Troubleshooting

### Common Issues

**No screenshots captured:**
```bash
# Check permissions (macOS)
python -c "from eidolon.core import Observer; Observer().test_permissions()"

# Check monitoring status
python -m eidolon status
```

**Search returns no results:**
```bash
# Check if data exists
ls -la data/screenshots/

# Rebuild search index
python -m eidolon rebuild-index
```

**High memory usage:**
```bash
# Reduce memory limits
python -m eidolon config set max_memory_mb 1024

# Clean up old data
python -m eidolon cleanup --days 7
```

### Performance Optimization

```bash
# Reduce capture frequency
python -m eidolon config set capture_interval 60

# Increase activity threshold
python -m eidolon config set activity_threshold 0.3

# Limit concurrent processing
python -m eidolon config set max_workers 2
```

## Integration Examples

### Shell Aliases

Add to your `.bashrc` or `.zshrc`:

```bash
alias esearch="python -m eidolon search"
alias echat="python -m eidolon chat"
alias estatus="python -m eidolon status"
alias ecleanup="python -m eidolon cleanup --days 30"
```

### Cron Jobs

Automate maintenance tasks:

```bash
# Daily cleanup (keep 30 days)
0 2 * * * cd /path/to/eidolon && python -m eidolon cleanup --days 30

# Weekly health check
0 9 * * 1 cd /path/to/eidolon && ./scripts/health_check.sh
```

### Keyboard Shortcuts

Set up system shortcuts for quick access:

- **⌘+Shift+E**: `python -m eidolon search`
- **⌘+Shift+C**: `python -m eidolon chat`
- **⌘+Shift+S**: `python -m eidolon status`

## Best Practices

1. **Regular Cleanup**: Run cleanup weekly to manage storage
2. **Monitor Performance**: Check system status regularly
3. **Backup Important Data**: Export search results for important projects
4. **Privacy Settings**: Review and adjust redaction patterns
5. **Update Regularly**: Keep dependencies updated for security

## Getting Help

- Run `python -m eidolon help` for command-specific help
- Check logs in `data/logs/` for detailed error information
- Use `./scripts/health_check.sh` to diagnose system issues
- Review the [Installation Guide](installation.md) for setup issues
