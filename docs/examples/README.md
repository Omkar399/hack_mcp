# Eidolon Examples

This directory contains practical examples and use cases for the Eidolon AI Personal Assistant.

## 📚 Available Examples

- **[Usage Examples](USAGE_EXAMPLES.md)** - Comprehensive usage scenarios
  - Basic screenshot monitoring
  - Advanced search techniques
  - AI-powered analysis
  - Natural language queries
  - Workflow automation
  - Real-world use cases

## 🚀 Quick Examples

### Basic Monitoring
```bash
# Start capturing screenshots
python -m eidolon capture

# Custom interval
python -m eidolon capture --interval 30
```

### Searching Activity
```bash
# Text search
python -m eidolon search "python code"

# Natural language query
python -m eidolon search "What did I work on yesterday?"
```

### AI Analysis
```bash
# Analyze recent captures
python -m eidolon analyze --recent 10

# Semantic search
python -m eidolon search "machine learning tutorials" --semantic
```

## 📊 Example Categories

### Development Workflows
- Monitoring coding sessions
- Tracking debugging progress
- Finding code snippets
- Analyzing productivity

### Research & Learning
- Capturing research materials
- Finding referenced articles
- Tracking learning progress
- Building knowledge base

### Communication
- Email activity tracking
- Meeting notes capture
- Communication patterns
- Contact history

### Productivity
- Time tracking
- Task completion
- Focus analysis
- Distraction patterns

## 💡 Best Practices

1. **Adjust capture intervals** based on activity type
2. **Use natural language** for complex queries
3. **Leverage AI features** for deeper insights
4. **Regular cleanup** to manage storage
5. **Privacy settings** for sensitive work

## 🔧 Advanced Examples

See [USAGE_EXAMPLES.md](USAGE_EXAMPLES.md) for detailed examples including:
- Python API usage
- Custom scripts
- Integration patterns
- Automation workflows
- Performance optimization

## 🆘 Need More Examples?

- Check the [test files](../../test_phase*.py) for code examples
- See [API documentation](../api/) for technical details
- Visit [GitHub Discussions](https://github.com/eidolon-ai/eidolon/discussions) for community examples