# Quick Start Guide

Get up and running with Eidolon AI Personal Assistant in just 5 minutes! This guide assumes you've already completed the [Installation Guide](installation.md).

## 🏁 First Launch

### 1. Verify Installation

```bash
# Check that Eidolon is properly installed
python -m eidolon --version

# Expected output: Eidolon v1.0.0 (or current version)
```

### 2. Initialize Your Environment

```bash
# Run the guided setup (recommended for first-time users)
python -m eidolon init --guided
```

This will:
- Create your configuration directory
- Set up default settings
- Create the initial database
- Configure basic privacy settings

### 3. Check System Status

```bash
# Verify everything is working correctly
python -m eidolon status
```

You should see output indicating all components are ready.

## 🎯 Core Workflow

### Start Monitoring (Background Mode)

```bash
# Start background monitoring
python -m eidolon capture --daemon

# Or run in foreground to see activity
python -m eidolon capture
```

Eidolon will now:
- Capture screenshots at intelligent intervals
- Extract text and UI elements using OCR
- Analyze content for importance
- Store everything in your local knowledge base

### Your First Search

After a few minutes of activity, try searching:

```bash
# Search for recent activity
python -m eidolon search "what did I work on today?"

# Search for specific content
python -m eidolon search "email" --time-range "last 2 hours"

# Search with filters
python -m eidolon search "meeting" --type "text" --limit 5
```

### Interactive Chat Mode

Launch the interactive chat interface:

```bash
# Start chat mode
python -m eidolon chat
```

Now you can ask natural language questions:
- "What projects have I been working on this week?"
- "Show me all the websites I visited about Python"
- "What emails did I receive today?"
- "Summarize my activity from yesterday"

## 🔧 Essential Configuration

### Privacy Settings

Configure what Eidolon monitors:

```bash
# Edit privacy settings
python -m eidolon config privacy

# Exclude specific applications
python -m eidolon config set privacy.excluded_apps "['1Password', 'Banking App']"

# Set sensitive patterns to redact
python -m eidolon config set privacy.sensitive_patterns "['password', 'ssn', 'credit card']"
```

### Performance Tuning

Adjust capture frequency based on your needs:

```bash
# Reduce frequency for better performance (capture every 30 seconds)
python -m eidolon config set observer.capture_interval 30

# Increase sensitivity for more captures
python -m eidolon config set observer.activity_threshold 0.03
```

### Cloud AI Integration (Optional)

For enhanced analysis capabilities, configure cloud APIs:

```bash
# Set OpenAI API key for advanced analysis
python -m eidolon config set analysis.cloud_apis.openai_key "your-api-key"

# Configure cost limits
python -m eidolon config set analysis.routing.cost_limit_daily 5.0
```

## 🚀 Common Use Cases

### 1. Track Work Sessions

```bash
# Tag the start of a work session
python -m eidolon tag "Starting React project work"

# Later, search for this session
python -m eidolon search "React project" --tagged
```

### 2. Find Lost Information

```bash
# Find that article you read yesterday
python -m eidolon search "machine learning article" --time-range "yesterday"

# Find a specific website you visited
python -m eidolon search "github.com/specific-repo" --type "url"
```

### 3. Productivity Analysis

```bash
# Get productivity insights
python -m eidolon insights --type "productivity" --period "week"

# Analyze time spent on different activities
python -m eidolon analyze "time distribution" --detailed
```

### 4. Meeting Preparation

```bash
# Find all mentions of a upcoming meeting
python -m eidolon search "project standup meeting"

# Get context for a client call
python -m eidolon search "client name" --time-range "last month" --summarize
```

## 📱 Web Interface (Optional)

For a graphical interface, start the web server:

```bash
# Start web interface
python -m eidolon serve --host localhost --port 8080
```

Then open `http://localhost:8080` in your browser for:
- Visual timeline of your activity
- Advanced search interface  
- Configuration management
- Data export tools

## 🔍 Monitoring Your System

### Check What's Being Captured

```bash
# View recent screenshots
python -m eidolon show recent --limit 10

# Check capture statistics
python -m eidolon stats capture
```

### Review Storage Usage

```bash
# Check storage usage
python -m eidolon stats storage

# Clean up old data (older than 30 days)
python -m eidolon cleanup --older-than "30 days"
```

### Monitor Performance

```bash
# Check system performance impact
python -m eidolon stats performance

# View memory usage
python -m eidolon stats memory
```

## 🛠️ Customization Tips

### Adjust Capture Settings

```bash
# Only capture during work hours (9 AM - 6 PM)
python -m eidolon config set observer.active_hours "9-18"

# Exclude weekends
python -m eidolon config set observer.active_days "['monday', 'tuesday', 'wednesday', 'thursday', 'friday']"
```

### Custom Search Filters

```bash
# Create a custom search for work-related content
python -m eidolon search "work OR project OR meeting" --save-as "work-filter"

# Use your saved filter
python -m eidolon search --filter "work-filter" --time-range "today"
```

### Set Up Automation

```bash
# Auto-tag certain types of content
python -m eidolon config set automation.tags.code "['github.com', 'stackoverflow.com', '.py', '.js']"

# Auto-export daily summaries
python -m eidolon config set automation.export.daily_summary true
```

## 🚨 Quick Troubleshooting

### Nothing Being Captured?

```bash
# Check if monitoring is active
python -m eidolon status

# Restart monitoring
python -m eidolon restart

# Check for permission issues
python -m eidolon doctor
```

### Search Returns No Results?

```bash
# Check if any data exists
python -m eidolon stats total

# Rebuild search index
python -m eidolon reindex

# Try a broader search
python -m eidolon search "*" --limit 5
```

### Performance Issues?

```bash
# Check system resource usage
python -m eidolon stats performance

# Reduce capture frequency
python -m eidolon config set observer.capture_interval 60

# Enable local-only mode
python -m eidolon config set privacy.local_only_mode true
```

## 🎓 Learning More

### Next Steps

1. **Explore Features**: Read about [Core Features](features/) in detail
2. **Advanced Configuration**: Learn about [Advanced Configuration](configuration/advanced.md)
3. **Privacy Controls**: Review [Privacy Settings](configuration/privacy.md)
4. **Tutorials**: Try the [Daily Workflows](tutorials/daily-workflows.md) tutorial

### Best Practices

- **Start Small**: Begin with default settings and adjust as needed
- **Review Privacy**: Regularly check what's being captured
- **Regular Cleanup**: Periodically clean old data to maintain performance
- **Backup Configuration**: Save your configuration settings
- **Monitor Resources**: Keep an eye on disk space and memory usage

## 📞 Getting Help

If you encounter any issues:

1. Check the [Troubleshooting Guide](troubleshooting/common-issues.md)
2. Run the diagnostic tool: `python -m eidolon doctor`
3. Check the logs: `python -m eidolon logs --tail 50`
4. Visit our [Support Resources](troubleshooting/support.md)

## 🔒 Privacy Reminder

Remember that Eidolon captures your screen activity. Key privacy points:

- **Local by Default**: All data stays on your machine unless you enable cloud features
- **You Control What's Captured**: Configure exclusions for sensitive apps
- **Automatic Redaction**: Sensitive patterns are automatically hidden
- **Easy Data Management**: Export or delete your data anytime

---

**Congratulations!** 🎉 

You're now ready to experience the power of Eidolon AI Personal Assistant. The system will learn and improve as it observes your digital activity, becoming an increasingly valuable assistant for your daily tasks.

**Pro Tip**: Let Eidolon run for a full day before judging its usefulness. The more data it has, the better it becomes at understanding your work patterns and providing helpful insights.