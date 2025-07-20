# Eidolon Integration Documentation

Welcome to the comprehensive integration guide for Eidolon AI Personal Assistant. This documentation covers all aspects of integrating Eidolon with external systems, services, and development environments.

## 📚 Table of Contents

### Core Integrations
- **[Model Context Protocol (MCP)](mcp/README.md)** - Claude Desktop and MCP integration
- **[Claude Desktop Setup](claude-desktop/setup.md)** - Complete Claude Desktop configuration
- **[REST API Integration](api/rest-api.md)** - HTTP API for external systems
- **[WebSocket Integration](api/websocket.md)** - Real-time communication

### Development Integrations
- **[VS Code Extension](development/vscode.md)** - IDE integration for developers
- **[Jupyter Integration](development/jupyter.md)** - Notebook environment integration
- **[Git Integration](development/git.md)** - Version control integration
- **[CI/CD Integration](development/cicd.md)** - Continuous integration pipelines

### Third-Party Services
- **[Cloud Storage](services/cloud-storage.md)** - AWS S3, Google Drive, OneDrive
- **[Communication Tools](services/communication.md)** - Slack, Teams, Discord
- **[Productivity Apps](services/productivity.md)** - Notion, Trello, Asana
- **[AI Services](services/ai-services.md)** - OpenAI, Anthropic, Google AI

### Enterprise Integrations
- **[SSO & Identity](enterprise/sso.md)** - Single Sign-On integration
- **[LDAP/Active Directory](enterprise/ldap.md)** - Directory service integration
- **[Monitoring Systems](enterprise/monitoring.md)** - Prometheus, Grafana, DataDog
- **[Compliance Tools](enterprise/compliance.md)** - Audit and compliance integration

### Custom Development
- **[Plugin Development](custom/plugins.md)** - Create custom plugins
- **[Custom Tools](custom/tools.md)** - Develop custom MCP tools
- **[Webhook Integration](custom/webhooks.md)** - Event-driven integrations
- **[SDK Usage](custom/sdk.md)** - Use Eidolon SDK in your applications

## 🎯 Quick Integration Guide

### 1. Claude Desktop Integration (5 minutes)

```bash
# Install Eidolon with MCP support
pip install eidolon-ai[mcp]

# Start MCP server
python -m eidolon mcp start --port 3001

# Add to Claude Desktop config
echo '{
  "mcpServers": {
    "eidolon": {
      "command": "python",
      "args": ["-m", "eidolon", "mcp", "serve"],
      "env": {
        "EIDOLON_CONFIG_PATH": "~/.eidolon/config"
      }
    }
  }
}' > ~/.claude/config.json

# Restart Claude Desktop
```

### 2. REST API Integration (10 minutes)

```python
import requests

# Start Eidolon API server
# python -m eidolon serve --api-only --port 8080

# Search your digital memory
response = requests.post('http://localhost:8080/api/v1/search', 
    json={
        'query': 'what did I work on yesterday?',
        'limit': 10,
        'time_range': 'yesterday'
    },
    headers={'Authorization': 'Bearer YOUR_API_KEY'}
)

results = response.json()
for item in results['data']:
    print(f"{item['timestamp']}: {item['content']}")
```

### 3. VS Code Extension (2 minutes)

```bash
# Install the Eidolon VS Code extension
code --install-extension eidolon-ai.eidolon-vscode

# Configure workspace settings
echo '{
  "eidolon.apiUrl": "http://localhost:8080",
  "eidolon.apiKey": "your-api-key",
  "eidolon.autoCapture": true
}' > .vscode/settings.json
```

## 🔧 Integration Architecture

### Integration Patterns

```
┌─────────────────────────────────────────────────────────────┐
│                External Applications                         │
│  Claude Desktop • VS Code • Slack • Notion • Custom Apps    │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                Integration Layer                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │     MCP     │ │  REST API   │ │  WebSocket  │           │
│  │   Server    │ │  Endpoints  │ │   Server    │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │   Webhooks  │ │   Plugins   │ │     SDK     │           │
│  │   Handler   │ │   System    │ │  Libraries  │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                   Eidolon Core                              │
│  Observer • Analyzer • Memory • Query Processor             │
└─────────────────────────────────────────────────────────────┘
```

### Integration Types

#### 1. **Pull Integrations**
- External systems query Eidolon data
- REST API endpoints
- GraphQL queries
- Database connections

#### 2. **Push Integrations**
- Eidolon sends data to external systems
- Webhooks
- Message queues
- Direct API calls

#### 3. **Bidirectional Integrations**
- Real-time synchronization
- WebSocket connections
- MCP protocol
- Plugin interfaces

#### 4. **Embedded Integrations**
- SDK libraries
- Browser extensions
- IDE plugins
- Mobile apps

## 🛠️ MCP Integration Deep Dive

### Model Context Protocol Overview

MCP enables seamless integration between Eidolon and Claude Desktop, allowing Claude to access your digital memory and perform actions on your behalf.

### Available MCP Tools

```json
{
  "tools": [
    {
      "name": "search_activity",
      "description": "Search through captured digital activity",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query": {"type": "string"},
          "time_range": {"type": "string"},
          "limit": {"type": "integer"}
        }
      }
    },
    {
      "name": "get_insights",
      "description": "Get productivity and behavior insights",
      "inputSchema": {
        "type": "object", 
        "properties": {
          "type": {"enum": ["productivity", "patterns", "summary"]},
          "period": {"type": "string"}
        }
      }
    },
    {
      "name": "capture_context",
      "description": "Capture current screen context",
      "inputSchema": {
        "type": "object",
        "properties": {
          "description": {"type": "string"}
        }
      }
    }
  ]
}
```

### MCP Resources

```json
{
  "resources": [
    {
      "uri": "eidolon://activity/recent",
      "name": "Recent Activity",
      "description": "Last 24 hours of digital activity"
    },
    {
      "uri": "eidolon://projects/current",
      "name": "Current Projects", 
      "description": "Active project contexts and progress"
    },
    {
      "uri": "eidolon://insights/productivity",
      "name": "Productivity Insights",
      "description": "Productivity patterns and recommendations"
    }
  ]
}
```

## 📡 REST API Reference

### Authentication

```bash
# API Key Authentication
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     http://localhost:8080/api/v1/search
```

### Core Endpoints

#### Search Activity
```http
POST /api/v1/search
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY

{
  "query": "Python development work",
  "time_range": "last_week",
  "filters": {
    "content_type": ["code", "documentation"],
    "applications": ["VSCode", "Terminal"]
  },
  "limit": 20
}
```

#### Get Insights
```http
GET /api/v1/insights?type=productivity&period=week
Authorization: Bearer YOUR_API_KEY
```

#### Create Memory
```http
POST /api/v1/memories
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY

{
  "content": "Important meeting notes about Q4 planning",
  "type": "note",
  "tags": ["meeting", "q4", "planning"],
  "metadata": {
    "source": "manual_entry",
    "importance": "high"
  }
}
```

## 🔌 Plugin Development

### Plugin Architecture

```python
from eidolon.plugins import BasePlugin, plugin_registry

class CustomAnalyzerPlugin(BasePlugin):
    name = "custom_analyzer"
    version = "1.0.0"
    
    def initialize(self, config):
        self.analyzer = CustomAnalyzer(config)
    
    async def process_screenshot(self, screenshot):
        """Custom screenshot analysis"""
        analysis = await self.analyzer.analyze(screenshot)
        return {
            'custom_insights': analysis.insights,
            'confidence': analysis.confidence
        }
    
    def get_tools(self):
        """MCP tools provided by this plugin"""
        return [
            {
                'name': 'custom_analysis',
                'description': 'Perform custom analysis',
                'handler': self.handle_custom_analysis
            }
        ]

# Register the plugin
plugin_registry.register(CustomAnalyzerPlugin)
```

### Plugin Installation

```bash
# Install plugin from PyPI
pip install eidolon-plugin-custom-analyzer

# Install plugin from local source
pip install -e ./my-eidolon-plugin

# Enable plugin in configuration
python -m eidolon config plugins.enable custom_analyzer
```

## 🌐 Webhook Integration

### Webhook Configuration

```yaml
# ~/.eidolon/config/webhooks.yaml
webhooks:
  slack_notifications:
    url: "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
    events: ["significant_activity", "productivity_alert"]
    headers:
      Authorization: "Bearer slack-token"
  
  custom_integration:
    url: "https://your-service.com/eidolon/webhook"
    events: ["*"]
    filters:
      importance_threshold: 0.8
```

### Webhook Events

```python
# Example webhook payload
{
  "event_type": "significant_activity",
  "timestamp": "2025-07-20T10:30:00Z",
  "data": {
    "activity_type": "code_development",
    "duration": 3600,
    "applications": ["VSCode", "Terminal", "Browser"],
    "insights": {
      "productivity_score": 0.85,
      "focus_time": 0.92,
      "context_switches": 3
    }
  },
  "metadata": {
    "user_id": "user123",
    "session_id": "session456"
  }
}
```

## 📱 Mobile Integration

### iOS Shortcuts Integration

```javascript
// Shortcut to search Eidolon from iOS
const query = await getInput("What would you like to search for?");
const response = await fetch('https://your-eidolon.com/api/v1/search', {
    method: 'POST',
    headers: {
        'Authorization': 'Bearer YOUR_API_KEY',
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        query: query,
        limit: 5
    })
});
const results = await response.json();
return results.data.map(item => item.content).join('\n\n');
```

### Android Tasker Integration

```bash
# Tasker HTTP Request task
URL: https://your-eidolon.com/api/v1/insights
Method: GET
Headers: Authorization: Bearer YOUR_API_KEY
Response: %eidolon_insights

# Use insights in notification
Notify: Today's Productivity: %eidolon_insights
```

## 🔧 SDK Usage

### Python SDK

```python
from eidolon_sdk import EidolonClient

# Initialize client
client = EidolonClient(
    base_url="http://localhost:8080",
    api_key="your-api-key"
)

# Search activities
results = await client.search(
    query="machine learning research",
    time_range="last_month"
)

# Get insights
insights = await client.get_insights(
    type="productivity",
    period="week"
)

# Stream real-time events
async for event in client.stream_events():
    if event.type == "significant_activity":
        print(f"New activity: {event.data}")
```

### JavaScript SDK

```javascript
import { EidolonClient } from '@eidolon-ai/sdk';

const client = new EidolonClient({
    baseUrl: 'http://localhost:8080',
    apiKey: 'your-api-key'
});

// Search with async/await
const results = await client.search({
    query: 'meeting notes',
    timeRange: 'today'
});

// Real-time updates
client.on('activity', (activity) => {
    console.log('New activity captured:', activity);
});
```

## 🏢 Enterprise Integration

### SSO Integration

```yaml
# SAML SSO Configuration
sso:
  provider: "saml"
  settings:
    entity_id: "eidolon"
    sso_url: "https://idp.company.com/sso/saml"
    x509_cert: "/path/to/idp-cert.pem"
    metadata_url: "https://idp.company.com/metadata"
```

### LDAP Integration

```yaml
# Active Directory/LDAP Configuration
ldap:
  server: "ldap://ad.company.com"
  bind_dn: "CN=eidolon,OU=Service Accounts,DC=company,DC=com"
  bind_password: "${LDAP_PASSWORD}"
  user_search:
    base_dn: "OU=Users,DC=company,DC=com"
    filter: "(sAMAccountName={username})"
  group_search:
    base_dn: "OU=Groups,DC=company,DC=com"
    filter: "(member={user_dn})"
```

## 📊 Monitoring Integration

### Prometheus Metrics

```python
# Custom metrics endpoint
from prometheus_client import Counter, Histogram, generate_latest

# Eidolon exposes these metrics
eidolon_screenshots_total = Counter('eidolon_screenshots_total', 'Total screenshots captured')
eidolon_analysis_duration = Histogram('eidolon_analysis_duration_seconds', 'Analysis processing time')
eidolon_search_queries = Counter('eidolon_search_queries_total', 'Total search queries')

@app.route('/metrics')
def metrics():
    return generate_latest()
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Eidolon Performance",
    "panels": [
      {
        "title": "Screenshot Capture Rate",
        "targets": [
          {
            "expr": "rate(eidolon_screenshots_total[5m])",
            "legendFormat": "Screenshots/sec"
          }
        ]
      },
      {
        "title": "Analysis Performance", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, eidolon_analysis_duration_seconds_bucket)",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

## 🚨 Troubleshooting Integrations

### Common Integration Issues

#### MCP Connection Problems
```bash
# Check MCP server status
python -m eidolon mcp status

# Test MCP tools
python -m eidolon mcp test-tools

# Debug MCP communication
python -m eidolon mcp debug --verbose
```

#### API Authentication Issues
```bash
# Validate API key
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8080/api/v1/health

# Generate new API key
python -m eidolon admin api-keys create --name "integration-key"
```

#### Performance Issues
```bash
# Check integration performance
python -m eidolon stats integrations

# Monitor API response times
python -m eidolon monitor api --real-time

# Optimize for high-volume integrations
python -m eidolon config set api.rate_limit.burst 1000
```

## 📚 Integration Examples

### Complete Integration Examples

Check out our comprehensive integration examples:

- **[Slack Bot Integration](examples/slack-bot.md)** - Complete Slack bot with Eidolon
- **[Notion Sync](examples/notion-sync.md)** - Sync insights to Notion database
- **[Chrome Extension](examples/chrome-extension.md)** - Browser extension integration
- **[Zapier Integration](examples/zapier.md)** - No-code automation workflows

---

This integration documentation provides comprehensive guidance for connecting Eidolon with virtually any external system, service, or application to enhance your digital productivity and AI assistance capabilities.