# Eidolon User Guide

Welcome to the Eidolon AI Personal Assistant user guide. This section provides everything you need to get started with Eidolon and make the most of its features.

## 📚 User Guide Contents

### Getting Started
- **[Installation Guide](INSTALL.md)** - Complete installation instructions
  - System requirements
  - Platform-specific setup
  - Dependency installation
  - Permission configuration
  - Troubleshooting

### Quick Start (Coming Soon)
- Basic usage tutorial
- First capture session
- Searching your activity
- Understanding the results

### Features Guide (Coming Soon)
- Screenshot monitoring
- Text extraction (OCR)
- AI-powered analysis
- Semantic search
- Natural language queries

### Advanced Usage (Coming Soon)
- Custom configurations
- Performance optimization
- Privacy settings
- Cloud AI setup
- Automation workflows

## 🚀 Quick Start Commands

### Basic Workflow
```bash
# 1. Activate virtual environment
source eidolon_env/bin/activate

# 2. Start monitoring
python -m eidolon capture

# 3. Search your activity
python -m eidolon search "python code"

# 4. Check status
python -m eidolon status
```

### Common Use Cases

#### Monitor Coding Sessions
```bash
python -m eidolon capture --interval 5
```

#### Search Recent Work
```bash
python -m eidolon search "function" --limit 10
```

#### Natural Language Queries (Phase 4)
```bash
python -m eidolon search "What Python code did I write yesterday?"
```

## 📊 Feature Overview

### Phase 1-2 Features (Available Now)
- ✅ Automatic screenshot capture
- ✅ OCR text extraction
- ✅ Content classification
- ✅ Full-text search
- ✅ Activity monitoring

### Phase 3-4 Features (Available Now)
- ✅ AI scene analysis
- ✅ Semantic search
- ✅ Natural language queries
- ✅ Cloud AI integration
- ✅ RAG responses

### Phase 5-7 Features (Coming Soon)
- 📋 Productivity analytics
- 📋 Pattern recognition
- 📋 Autonomous actions
- 📋 Digital twin capabilities

## 🔧 Configuration

Eidolon can be configured through:
1. **Configuration file**: `config/settings.yaml`
2. **Environment variables**: For API keys and secrets
3. **Command-line arguments**: For runtime options

### Key Settings
```yaml
observer:
  capture_interval: 10      # Seconds between captures
  activity_threshold: 0.05  # Change detection sensitivity

analysis:
  local_models:
    vision: "microsoft/florence-2-base"
  cloud_apis:
    gemini_key: "${GEMINI_API_KEY}"
```

## 🔒 Privacy & Security

Eidolon is designed with privacy in mind:
- **Local-first**: All processing happens on your machine by default
- **User control**: You decide what goes to cloud APIs
- **Auto-redaction**: Sensitive information is automatically hidden
- **Excluded apps**: Password managers are never monitored
- **Data ownership**: Your data stays yours

## 💡 Tips & Best Practices

### Optimize Performance
1. Adjust capture interval based on activity
2. Use higher intervals for reading/research
3. Use lower intervals for active work
4. Monitor resource usage with `status` command

### Improve Search Results
1. Use specific keywords
2. Try natural language queries
3. Filter by content type
4. Use time-based filters

### Manage Storage
1. Regular cleanup of old data
2. Export important findings
3. Monitor disk usage
4. Adjust retention policies

## 🆘 Getting Help

### Troubleshooting
- Check [Installation Guide](INSTALL.md) for common issues
- Run `python validate_dependencies.py` to check setup
- Review logs in `logs/eidolon.log`

### Support Resources
- **Documentation**: You're here!
- **GitHub Issues**: Report bugs and request features
- **Discussions**: Community support
- **Examples**: See [Usage Examples](../examples/USAGE_EXAMPLES.md)

## 🎯 Next Steps

1. Complete the [installation](INSTALL.md)
2. Run your first capture session
3. Try searching your activity
4. Explore advanced features
5. Customize your configuration

Welcome to your AI-powered digital memory!