# Eidolon AI Personal Assistant - Usage Examples and Tutorials

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Usage Examples](#basic-usage-examples)
3. [Advanced Workflows](#advanced-workflows)
4. [Integration Patterns](#integration-patterns)
5. [Custom Extensions](#custom-extensions)
6. [Real-World Use Cases](#real-world-use-cases)
7. [Performance Optimization](#performance-optimization)
8. [Best Practices](#best-practices)

## Getting Started

### Quick Start Tutorial

This tutorial will walk you through setting up Eidolon and capturing your first screenshots with AI analysis.

#### Step 1: Installation and Setup

```bash
# Clone and install
git clone https://github.com/your-org/eidolon.git
cd eidolon

# Create virtual environment
python -m venv eidolon_env
source eidolon_env/bin/activate  # On Windows: eidolon_env\Scripts\activate

# Install dependencies
pip install -e .

# Verify installation
python -m eidolon version
```

#### Step 2: Basic Configuration

Create your configuration file:

```yaml
# eidolon/config/my_settings.yaml
observer:
  capture_interval: 30          # Capture every 30 seconds
  storage_path: "./my_data"     # Your data directory
  
analysis:
  local_models:
    vision: "microsoft/florence-2-base"
  routing:
    local_first: true           # Use local models first

privacy:
  excluded_apps:               # Apps to exclude from monitoring
    - "1Password"
    - "Banking App"
    
logging:
  level: "INFO"
  file_path: "./logs/eidolon.log"
```

#### Step 3: Start Monitoring

```bash
# Start basic monitoring
python -m eidolon capture

# Or with custom config
python -m eidolon --config eidolon/config/my_settings.yaml capture

# Run in background
python -m eidolon capture --background
```

#### Step 4: Search Your Data

```bash
# Basic text search
python -m eidolon search "python code"

# Advanced search with filters
python -m eidolon search "meeting notes" --content-type document --limit 10

# Export results to JSON
python -m eidolon search "project" --format json > search_results.json
```

## Basic Usage Examples

### Example 1: Simple Screenshot Monitoring

Monitor your screen activity with basic analysis:

```python
from eidolon.core.observer import Observer
from eidolon.core.analyzer import Analyzer
import time

# Initialize components
observer = Observer()
analyzer = Analyzer()

# Start monitoring
observer.start_monitoring()
print("📸 Started monitoring...")

try:
    # Let it run for a minute
    time.sleep(60)
    
    # Check status
    status = observer.get_status()
    print(f"Captured {status['capture_count']} screenshots")
    
finally:
    # Stop monitoring
    observer.stop_monitoring()
    print("🛑 Stopped monitoring")
```

### Example 2: Manual Screenshot Analysis

Capture and analyze specific screenshots:

```python
from eidolon.core.observer import Observer
from eidolon.core.analyzer import Analyzer
from pathlib import Path

# Initialize
observer = Observer()
analyzer = Analyzer()

# Capture screenshot
screenshot = observer.capture_screenshot()
print(f"📷 Captured: {screenshot.width}x{screenshot.height}")

# Extract text
extracted_text = analyzer.extract_text(screenshot.file_path)
print(f"📝 Extracted text: {extracted_text.word_count} words")
print(f"Text preview: {extracted_text.text[:100]}...")

# Analyze content
content_analysis = analyzer.analyze_content(screenshot.file_path, extracted_text.text)
print(f"🏷️  Content type: {content_analysis.content_type}")
print(f"🎯 Confidence: {content_analysis.confidence:.2f}")
print(f"🏷️  Tags: {', '.join(content_analysis.tags)}")
```

### Example 3: Semantic Search

Search your captured content using natural language:

```python
import asyncio
from eidolon.core.memory import MemorySystem

async def semantic_search_example():
    memory = MemorySystem()
    
    # Search for development-related content
    results = await memory.semantic_search(
        query="Python programming and debugging",
        n_results=5
    )
    
    print("🔍 Search results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Similarity: {result.similarity:.2f}")
        print(f"   Content: {result.content[:100]}...")
        print(f"   Time: {result.metadata.get('timestamp', 'Unknown')}")
        print()

# Run the search
asyncio.run(semantic_search_example())
```

### Example 4: Content Classification

Automatically classify different types of screen content:

```python
from eidolon.core.analyzer import Analyzer
from pathlib import Path

analyzer = Analyzer()

# Test different content types
test_images = [
    "screenshots/code_editor.png",
    "screenshots/web_browser.png", 
    "screenshots/document.png"
]

for image_path in test_images:
    if Path(image_path).exists():
        # Extract and analyze
        text = analyzer.extract_text(image_path)
        analysis = analyzer.analyze_content(image_path, text.text)
        
        print(f"📁 {image_path}")
        print(f"   Type: {analysis.content_type}")
        print(f"   Tags: {analysis.tags}")
        print(f"   Description: {analysis.description}")
        print()
```

## Advanced Workflows

### Workflow 1: Project Timeline Reconstruction

Track your work on different projects over time:

```python
import asyncio
from datetime import datetime, timedelta
from eidolon.core.query_processor import QueryProcessor

async def project_timeline():
    processor = QueryProcessor()
    
    # Define time range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    # Query for project timeline
    query = {
        "intent": "timeline",
        "temporal": {
            "start_date": start_date,
            "end_date": end_date
        },
        "entities": [
            {"type": "application", "value": "vscode"},
            {"type": "quoted_text", "value": "python"}
        ]
    }
    
    result = await processor.process_query(query, limit=50)
    
    print("📅 Project Timeline (Last 7 days):")
    for event in result.data:
        timestamp = event.get('timestamp', 'Unknown')
        app = event.get('application', 'Unknown')
        title = event.get('title', 'No title')
        
        print(f"⏰ {timestamp} - {app}: {title}")

asyncio.run(project_timeline())
```

### Workflow 2: Productivity Analytics

Analyze your productivity patterns:

```python
import asyncio
from eidolon.core.analytics import AnalyticsEngine
from datetime import datetime, timedelta

async def productivity_analysis():
    analytics = AnalyticsEngine()
    
    # Analyze last month
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Get productivity summary
    summary = await analytics.get_analytics_summary(start_date, end_date)
    
    print("📊 Productivity Summary (Last 30 days):")
    print(f"📱 Active applications: {len(summary.get('applications_used', []))}")
    print(f"⏱️  Total active time: {summary.get('total_active_time_hours', 0):.1f} hours")
    print(f"📈 Most productive day: {summary.get('most_productive_day', 'Unknown')}")
    
    # Application usage breakdown
    print("\n🎯 Top Applications:")
    for app, hours in summary.get('app_usage_hours', {}).items():
        print(f"   {app}: {hours:.1f} hours")

asyncio.run(productivity_analysis())
```

### Workflow 3: Intelligent Content Organization

Automatically organize screenshots by content type and project:

```python
import asyncio
from eidolon.storage.metadata_db import MetadataDatabase
from eidolon.core.analyzer import Analyzer
from collections import defaultdict

async def organize_content():
    db = MetadataDatabase()
    analyzer = Analyzer()
    
    # Get all screenshots from last week
    screenshots = db.get_recent_screenshots(days=7)
    
    # Organize by content type
    organized = defaultdict(list)
    
    for screenshot in screenshots:
        # Get analysis for this screenshot
        analysis = db.get_content_analysis(screenshot['id'])
        
        if analysis:
            content_type = analysis.get('content_type', 'unknown')
            organized[content_type].append({
                'timestamp': screenshot['timestamp'],
                'description': analysis.get('description', 'No description'),
                'tags': analysis.get('tags', [])
            })
    
    # Display organization
    print("📂 Content Organization (Last 7 days):")
    for content_type, items in organized.items():
        print(f"\n📁 {content_type.title()} ({len(items)} items)")
        for item in items[:3]:  # Show first 3
            print(f"   • {item['timestamp']}: {item['description'][:50]}...")
        if len(items) > 3:
            print(f"   ... and {len(items) - 3} more")

asyncio.run(organize_content())
```

### Workflow 4: Smart Notifications

Get notified about important content patterns:

```python
import asyncio
from eidolon.core.memory import MemorySystem
from eidolon.storage.metadata_db import MetadataDatabase

async def smart_notifications():
    memory = MemorySystem()
    db = MetadataDatabase()
    
    # Define notification triggers
    triggers = [
        "error message",
        "build failed",
        "pull request",
        "meeting invite",
        "deadline"
    ]
    
    print("🔔 Smart Notifications:")
    
    for trigger in triggers:
        # Search for recent matches
        results = await memory.semantic_search(
            query=trigger,
            n_results=3
        )
        
        # Filter for recent results (last 2 hours)
        recent_results = [
            r for r in results 
            if r.similarity > 0.8  # High confidence
        ]
        
        if recent_results:
            print(f"\n⚠️  Alert: Found {len(recent_results)} instances of '{trigger}'")
            for result in recent_results:
                timestamp = result.metadata.get('timestamp', 'Unknown')
                print(f"   📅 {timestamp}: {result.content[:80]}...")

asyncio.run(smart_notifications())
```

## Integration Patterns

### Pattern 1: Web API Integration

Create a REST API for external access:

```python
from fastapi import FastAPI, HTTPException
from eidolon.core.memory import MemorySystem
from eidolon.core.observer import Observer
import asyncio

app = FastAPI(title="Eidolon API")
memory = MemorySystem()
observer = Observer()

@app.post("/search")
async def search_content(query: str, limit: int = 10):
    """Search captured content."""
    try:
        results = await memory.semantic_search(query, n_results=limit)
        return {
            "query": query,
            "results": [
                {
                    "content": r.content,
                    "similarity": r.similarity,
                    "timestamp": r.metadata.get("timestamp")
                }
                for r in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Get system status."""
    return observer.get_status()

@app.post("/capture")
async def manual_capture():
    """Capture screenshot manually."""
    try:
        screenshot = observer.capture_screenshot()
        return {
            "success": True,
            "screenshot_id": screenshot.id,
            "timestamp": screenshot.timestamp.isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn api:app --host 0.0.0.0 --port 8000
```

### Pattern 2: Slack Bot Integration

Create a Slack bot for querying your data:

```python
import asyncio
from slack_bolt.async_app import AsyncApp
from eidolon.core.memory import MemorySystem

app = AsyncApp(token="your-slack-bot-token")
memory = MemorySystem()

@app.message("search")
async def handle_search(message, say):
    """Handle search requests in Slack."""
    query_text = message['text'].replace('search', '').strip()
    
    if not query_text:
        await say("Please provide a search query. Example: `search python code`")
        return
    
    try:
        # Search content
        results = await memory.semantic_search(query_text, n_results=5)
        
        if results:
            response = f"🔍 Found {len(results)} results for '{query_text}':\n\n"
            for i, result in enumerate(results, 1):
                timestamp = result.metadata.get('timestamp', 'Unknown')
                response += f"{i}. _{timestamp}_\n{result.content[:100]}...\n\n"
        else:
            response = f"No results found for '{query_text}'"
        
        await say(response)
        
    except Exception as e:
        await say(f"Search failed: {str(e)}")

@app.command("/eidolon_status")
async def handle_status(ack, respond):
    """Get Eidolon status via Slack command."""
    await ack()
    
    # Get status (implement Observer status check)
    status = {"running": True, "capture_count": 42}  # Placeholder
    
    response = f"""
📊 *Eidolon Status*
Running: {"✅" if status['running'] else "❌"}
Captures: {status.get('capture_count', 0)}
    """
    
    await respond(response)

# Run with: python slack_bot.py
if __name__ == "__main__":
    asyncio.run(app.async_start(port=3000))
```

### Pattern 3: Home Assistant Integration

Integrate with Home Assistant for smart home automation:

```python
import asyncio
import aiohttp
from eidolon.core.memory import MemorySystem

class HomeAssistantIntegration:
    def __init__(self, ha_url: str, ha_token: str):
        self.ha_url = ha_url
        self.ha_token = ha_token
        self.memory = MemorySystem()
        
    async def notify_work_session_start(self):
        """Notify HA when work session starts."""
        # Detect work session start (e.g., IDE opened)
        results = await self.memory.semantic_search("code editor opened", n_results=1)
        
        if results and results[0].similarity > 0.9:
            # Send to Home Assistant
            await self._call_ha_service(
                "notify.mobile_app",
                {"message": "Work session started - adjusting environment"}
            )
            
            # Adjust smart home settings
            await self._call_ha_service(
                "light.turn_on",
                {"entity_id": "light.desk_lamp", "brightness": 200}
            )
            
    async def _call_ha_service(self, service: str, data: dict):
        """Call Home Assistant service."""
        headers = {
            "Authorization": f"Bearer {self.ha_token}",
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            await session.post(
                f"{self.ha_url}/api/services/{service.replace('.', '/')}",
                headers=headers,
                json=data
            )

# Usage
async def main():
    ha = HomeAssistantIntegration("http://homeassistant:8123", "your-ha-token")
    await ha.notify_work_session_start()

asyncio.run(main())
```

## Custom Extensions

### Extension 1: Custom Content Analyzer

Create a specialized analyzer for your domain:

```python
from eidolon.core.analyzer import Analyzer, ContentAnalysis
from typing import List
import re

class CustomAnalyzer(Analyzer):
    """Specialized analyzer for software development workflows."""
    
    def __init__(self):
        super().__init__()
        
        # Custom patterns for development content
        self.dev_patterns = {
            'git_command': r'git\s+(add|commit|push|pull|checkout|merge)',
            'error_log': r'(error|exception|traceback|failed)',
            'test_results': r'(passed|failed|skipped).*tests?',
            'build_status': r'(build\s+(passed|failed|success))',
            'code_review': r'(pull\s+request|merge\s+request|review)',
        }
    
    def extract_development_context(self, text: str) -> dict:
        """Extract development-specific context from text."""
        context = {
            'commands': [],
            'errors': [],
            'test_info': {},
            'build_status': None
        }
        
        for pattern_name, pattern in self.dev_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            
            for match in matches:
                if pattern_name == 'git_command':
                    context['commands'].append(match.group())
                elif pattern_name == 'error_log':
                    context['errors'].append(match.group())
                elif pattern_name == 'test_results':
                    # Extract test numbers
                    numbers = re.findall(r'\d+', match.group())
                    if numbers:
                        context['test_info']['count'] = int(numbers[0])
                elif pattern_name == 'build_status':
                    context['build_status'] = match.group()
        
        return context
    
    def analyze_content(self, image_path, extracted_text: str = "") -> ContentAnalysis:
        """Enhanced content analysis with development context."""
        # Get base analysis
        base_analysis = super().analyze_content(image_path, extracted_text)
        
        # Add development context
        dev_context = self.extract_development_context(extracted_text)
        
        # Enhanced tags
        enhanced_tags = base_analysis.tags.copy()
        
        if dev_context['commands']:
            enhanced_tags.extend(['git', 'version-control'])
        if dev_context['errors']:
            enhanced_tags.extend(['debugging', 'error'])
        if dev_context['test_info']:
            enhanced_tags.append('testing')
        if dev_context['build_status']:
            enhanced_tags.append('build')
        
        # Enhanced description
        enhanced_description = base_analysis.description
        if dev_context['commands']:
            enhanced_description += f" Git commands: {', '.join(dev_context['commands'][:2])}"
        
        # Return enhanced analysis
        return ContentAnalysis(
            content_type=base_analysis.content_type,
            confidence=base_analysis.confidence,
            tags=list(set(enhanced_tags)),  # Remove duplicates
            description=enhanced_description,
            vision_analysis=base_analysis.vision_analysis
        )

# Usage example
analyzer = CustomAnalyzer()
analysis = analyzer.analyze_content("terminal_screenshot.png", "git commit -m 'fix bug'")
print(f"Enhanced tags: {analysis.tags}")
```

### Extension 2: Custom Storage Backend

Implement a custom storage backend:

```python
import asyncio
from typing import Dict, List, Any, Optional
from eidolon.storage.vector_db import VectorDatabase

class ElasticsearchVectorDB(VectorDatabase):
    """Custom Elasticsearch backend for vector storage."""
    
    def __init__(self, es_url: str, index_name: str = "eidolon_vectors"):
        # Don't call super().__init__() to avoid ChromaDB initialization
        self.es_url = es_url
        self.index_name = index_name
        self.client = None
        
    async def initialize(self):
        """Initialize Elasticsearch client."""
        try:
            from elasticsearch import AsyncElasticsearch
            self.client = AsyncElasticsearch([self.es_url])
            
            # Create index if it doesn't exist
            await self._create_index()
            
        except ImportError:
            raise ImportError("elasticsearch package required for ElasticsearchVectorDB")
    
    async def _create_index(self):
        """Create the Elasticsearch index with vector mapping."""
        mapping = {
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "dense_vector",
                        "dims": 384  # Adjust based on your embedding model
                    },
                    "content": {"type": "text"},
                    "screenshot_id": {"type": "keyword"},
                    "timestamp": {"type": "date"},
                    "metadata": {"type": "object"}
                }
            }
        }
        
        await self.client.indices.create(
            index=self.index_name,
            body=mapping,
            ignore=400  # Ignore if index already exists
        )
    
    async def store_content(
        self, 
        screenshot_id: str, 
        content_analysis: Dict[str, Any],
        extracted_text: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store content with vector embedding in Elasticsearch."""
        try:
            # Generate embedding (implement your embedding logic)
            embedding = await self._generate_embedding(content_analysis, extracted_text)
            
            doc = {
                "vector": embedding,
                "content": extracted_text,
                "screenshot_id": screenshot_id,
                "timestamp": metadata.get("timestamp") if metadata else None,
                "metadata": metadata or {}
            }
            
            await self.client.index(
                index=self.index_name,
                body=doc
            )
            
            return True
            
        except Exception as e:
            print(f"Failed to store in Elasticsearch: {e}")
            return False
    
    async def semantic_search(
        self,
        query: str,
        n_results: int = 10,
        content_type_filter: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using Elasticsearch kNN."""
        try:
            # Generate query embedding
            query_embedding = await self._generate_embedding({"description": query}, "")
            
            # Build search query
            search_body = {
                "size": n_results,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                            "params": {"query_vector": query_embedding}
                        }
                    }
                }
            }
            
            # Add filters if specified
            if content_type_filter:
                search_body["query"]["script_score"]["query"] = {
                    "term": {"metadata.content_type": content_type_filter}
                }
            
            response = await self.client.search(
                index=self.index_name,
                body=search_body
            )
            
            # Format results
            results = []
            for hit in response["hits"]["hits"]:
                results.append({
                    "id": hit["_id"],
                    "content": hit["_source"]["content"],
                    "similarity": hit["_score"] - 1.0,  # Adjust score
                    "metadata": hit["_source"]["metadata"]
                })
            
            return results
            
        except Exception as e:
            print(f"Elasticsearch search failed: {e}")
            return []

# Usage
async def use_custom_storage():
    # Initialize custom storage
    es_db = ElasticsearchVectorDB("http://localhost:9200")
    await es_db.initialize()
    
    # Store content
    await es_db.store_content(
        screenshot_id="test_123",
        content_analysis={"description": "Python code example"},
        extracted_text="def hello(): print('world')"
    )
    
    # Search
    results = await es_db.semantic_search("Python function")
    print(f"Found {len(results)} results")

asyncio.run(use_custom_storage())
```

## Real-World Use Cases

### Use Case 1: Software Development Assistant

Monitor your development workflow and provide intelligent assistance:

```python
import asyncio
from datetime import datetime, timedelta
from eidolon.core.observer import Observer
from eidolon.core.memory import MemorySystem

class DevAssistant:
    def __init__(self):
        self.observer = Observer()
        self.memory = MemorySystem()
        self.active_session = None
        
    async def start_dev_session(self, project_name: str):
        """Start a development session with tracking."""
        self.active_session = {
            "project": project_name,
            "start_time": datetime.now(),
            "activities": []
        }
        
        # Start monitoring
        self.observer.start_monitoring()
        print(f"🚀 Started dev session for {project_name}")
        
    async def end_dev_session(self):
        """End development session and generate summary."""
        if not self.active_session:
            return
        
        # Stop monitoring
        self.observer.stop_monitoring()
        
        # Generate session summary
        session_duration = datetime.now() - self.active_session["start_time"]
        
        # Search for session content
        results = await self.memory.semantic_search(
            f"project {self.active_session['project']} code development",
            n_results=20
        )
        
        # Analyze session
        summary = {
            "project": self.active_session["project"],
            "duration": session_duration,
            "activities_found": len(results),
            "focus_areas": self._extract_focus_areas(results)
        }
        
        print(f"📊 Session Summary:")
        print(f"   Project: {summary['project']}")
        print(f"   Duration: {summary['duration']}")
        print(f"   Activities: {summary['activities_found']}")
        print(f"   Focus areas: {', '.join(summary['focus_areas'])}")
        
        self.active_session = None
        return summary
        
    def _extract_focus_areas(self, results):
        """Extract main focus areas from session content."""
        focus_areas = set()
        
        for result in results:
            # Extract programming languages
            content = result.content.lower()
            if 'python' in content:
                focus_areas.add('Python')
            if 'javascript' in content or 'js' in content:
                focus_areas.add('JavaScript')
            if 'react' in content:
                focus_areas.add('React')
            # Add more patterns as needed
                
        return list(focus_areas)

# Usage
async def dev_workflow():
    assistant = DevAssistant()
    
    # Start session
    await assistant.start_dev_session("web-app-refactor")
    
    # Simulate work (in real usage, just work normally)
    print("💻 Working on project... (simulate by doing actual work)")
    await asyncio.sleep(10)  # In real usage, remove this
    
    # End session
    summary = await assistant.end_dev_session()

asyncio.run(dev_workflow())
```

### Use Case 2: Meeting Notes Extractor

Automatically extract and organize meeting notes from screen captures:

```python
import asyncio
import re
from datetime import datetime, timedelta
from eidolon.core.memory import MemorySystem
from eidolon.storage.metadata_db import MetadataDatabase

class MeetingNotesExtractor:
    def __init__(self):
        self.memory = MemorySystem()
        self.db = MetadataDatabase()
        
    async def extract_meeting_notes(self, date_range_days: int = 1):
        """Extract meeting notes from recent screenshots."""
        
        # Search for meeting-related content
        meeting_queries = [
            "meeting agenda",
            "action items",
            "zoom call",
            "teams meeting",
            "meeting notes",
            "follow up",
            "next steps"
        ]
        
        all_meeting_content = []
        
        for query in meeting_queries:
            results = await self.memory.semantic_search(query, n_results=10)
            
            # Filter for high confidence results from recent timeframe
            for result in results:
                if result.similarity > 0.7:  # High confidence
                    timestamp_str = result.metadata.get('timestamp', '')
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str)
                        if (datetime.now() - timestamp).days <= date_range_days:
                            all_meeting_content.append({
                                'query': query,
                                'content': result.content,
                                'timestamp': timestamp,
                                'similarity': result.similarity
                            })
                    except:
                        continue
        
        # Organize and summarize
        organized_notes = self._organize_meeting_content(all_meeting_content)
        return organized_notes
    
    def _organize_meeting_content(self, content_list):
        """Organize meeting content by type and time."""
        organized = {
            'action_items': [],
            'agenda_items': [],
            'decisions': [],
            'follow_ups': [],
            'other': []
        }
        
        for item in content_list:
            content = item['content'].lower()
            
            # Classify content type
            if any(word in content for word in ['action', 'todo', 'task', 'assign']):
                organized['action_items'].append(item)
            elif any(word in content for word in ['agenda', 'discuss', 'topic']):
                organized['agenda_items'].append(item)
            elif any(word in content for word in ['decided', 'agreed', 'resolution']):
                organized['decisions'].append(item)
            elif any(word in content for word in ['follow up', 'next', 'later']):
                organized['follow_ups'].append(item)
            else:
                organized['other'].append(item)
        
        return organized
    
    async def generate_meeting_summary(self, date_range_days: int = 1):
        """Generate a comprehensive meeting summary."""
        notes = await self.extract_meeting_notes(date_range_days)
        
        print(f"📝 Meeting Summary (Last {date_range_days} day(s)):")
        print("=" * 50)
        
        for category, items in notes.items():
            if items:
                print(f"\n📋 {category.replace('_', ' ').title()}:")
                for i, item in enumerate(items[:5], 1):  # Limit to 5 items
                    timestamp = item['timestamp'].strftime('%H:%M')
                    content_preview = item['content'][:80] + "..." if len(item['content']) > 80 else item['content']
                    print(f"   {i}. [{timestamp}] {content_preview}")
                
                if len(items) > 5:
                    print(f"   ... and {len(items) - 5} more items")
        
        return notes

# Usage
async def extract_todays_meetings():
    extractor = MeetingNotesExtractor()
    summary = await extractor.generate_meeting_summary(date_range_days=1)
    
    # Optional: Save to file
    with open(f"meeting_summary_{datetime.now().strftime('%Y%m%d')}.txt", 'w') as f:
        f.write(f"Meeting Summary - {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write("=" * 50 + "\n")
        
        for category, items in summary.items():
            if items:
                f.write(f"\n{category.replace('_', ' ').title()}:\n")
                for item in items:
                    f.write(f"- {item['content']}\n")

asyncio.run(extract_todays_meetings())
```

### Use Case 3: Research Assistant

Track and organize research activities across different topics:

```python
import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from eidolon.core.memory import MemorySystem

class ResearchAssistant:
    def __init__(self):
        self.memory = MemorySystem()
        self.research_topics = set()
        
    async def track_research_topic(self, topic: str, keywords: list):
        """Start tracking a research topic with associated keywords."""
        self.research_topics.add(topic)
        
        # Store the topic-keyword mapping
        topic_data = {
            "topic": topic,
            "keywords": keywords,
            "start_date": datetime.now().isoformat(),
            "status": "active"
        }
        
        print(f"🔬 Now tracking research topic: {topic}")
        print(f"📋 Keywords: {', '.join(keywords)}")
        
    async def analyze_research_progress(self, topic: str, days_back: int = 7):
        """Analyze research progress for a specific topic."""
        
        # Get topic keywords (in real implementation, store these)
        topic_keywords = {
            "machine_learning": ["neural network", "tensorflow", "pytorch", "ML", "deep learning"],
            "web_development": ["react", "javascript", "css", "html", "frontend", "backend"],
            "data_science": ["pandas", "numpy", "jupyter", "data analysis", "visualization"]
        }.get(topic, [topic])
        
        research_findings = defaultdict(list)
        
        # Search for each keyword
        for keyword in topic_keywords:
            results = await self.memory.semantic_search(keyword, n_results=20)
            
            for result in results:
                if result.similarity > 0.6:  # Medium confidence threshold
                    timestamp_str = result.metadata.get('timestamp', '')
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str)
                        if (datetime.now() - timestamp).days <= days_back:
                            research_findings[keyword].append({
                                'content': result.content,
                                'timestamp': timestamp,
                                'similarity': result.similarity
                            })
                    except:
                        continue
        
        # Generate research summary
        return self._generate_research_summary(topic, research_findings, days_back)
    
    def _generate_research_summary(self, topic: str, findings: dict, days_back: int):
        """Generate a research progress summary."""
        total_findings = sum(len(items) for items in findings.values())
        
        summary = {
            'topic': topic,
            'period': f"Last {days_back} days",
            'total_findings': total_findings,
            'keywords_found': len([k for k, v in findings.items() if v]),
            'most_active_keyword': max(findings.keys(), key=lambda k: len(findings[k])) if findings else None,
            'daily_activity': self._calculate_daily_activity(findings),
            'key_insights': self._extract_key_insights(findings)
        }
        
        return summary
    
    def _calculate_daily_activity(self, findings: dict):
        """Calculate research activity by day."""
        daily_counts = defaultdict(int)
        
        for keyword, items in findings.items():
            for item in items:
                day = item['timestamp'].date()
                daily_counts[day] += 1
        
        return dict(daily_counts)
    
    def _extract_key_insights(self, findings: dict):
        """Extract key insights from research findings."""
        insights = []
        
        for keyword, items in findings.items():
            if len(items) >= 3:  # Significant activity
                insights.append(f"High activity around '{keyword}' ({len(items)} instances)")
        
        return insights
    
    async def generate_research_report(self, topic: str):
        """Generate a comprehensive research report."""
        summary = await self.analyze_research_progress(topic)
        
        print(f"📊 Research Report: {summary['topic']}")
        print("=" * 50)
        print(f"📅 Period: {summary['period']}")
        print(f"🔍 Total findings: {summary['total_findings']}")
        print(f"🏷️  Active keywords: {summary['keywords_found']}")
        
        if summary['most_active_keyword']:
            print(f"🎯 Most researched: {summary['most_active_keyword']}")
        
        print(f"\n📈 Daily Activity:")
        for date, count in sorted(summary['daily_activity'].items()):
            print(f"   {date}: {count} research items")
        
        if summary['key_insights']:
            print(f"\n💡 Key Insights:")
            for insight in summary['key_insights']:
                print(f"   • {insight}")
        
        return summary

# Usage
async def research_workflow():
    assistant = ResearchAssistant()
    
    # Start tracking research topics
    await assistant.track_research_topic(
        "machine_learning",
        ["neural network", "tensorflow", "pytorch", "deep learning"]
    )
    
    # Simulate some research time
    print("\n🔍 Conducting research... (in real usage, browse/read normally)")
    await asyncio.sleep(2)
    
    # Generate research report
    report = await assistant.generate_research_report("machine_learning")

asyncio.run(research_workflow())
```

## Performance Optimization

### Optimization 1: Efficient Batch Processing

Process multiple screenshots efficiently:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from eidolon.core.analyzer import Analyzer
from eidolon.storage.metadata_db import MetadataDatabase
import time

class BatchProcessor:
    def __init__(self, max_workers: int = 4):
        self.analyzer = Analyzer()
        self.db = MetadataDatabase()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    async def process_screenshot_batch(self, screenshot_paths: list):
        """Process multiple screenshots in parallel."""
        start_time = time.time()
        
        # Create tasks for parallel processing
        tasks = []
        for path in screenshot_paths:
            task = asyncio.create_task(
                self._process_single_screenshot(path)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successful vs failed
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - successful
        
        processing_time = time.time() - start_time
        
        print(f"📊 Batch Processing Results:")
        print(f"   Total: {len(screenshot_paths)} screenshots")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        print(f"   Time: {processing_time:.2f} seconds")
        print(f"   Rate: {len(screenshot_paths)/processing_time:.1f} screenshots/second")
        
        return results
    
    async def _process_single_screenshot(self, path: str):
        """Process a single screenshot asynchronously."""
        try:
            # Run CPU-intensive tasks in thread pool
            loop = asyncio.get_event_loop()
            
            # Extract text
            extracted_text = await loop.run_in_executor(
                self.executor,
                self.analyzer.extract_text,
                path
            )
            
            # Analyze content
            content_analysis = await loop.run_in_executor(
                self.executor,
                self.analyzer.analyze_content,
                path,
                extracted_text.text
            )
            
            # Store results
            screenshot_id = self.db.store_screenshot({
                "file_path": path,
                "timestamp": time.time()
            })
            
            if extracted_text.text:
                self.db.store_ocr_result(screenshot_id, extracted_text.to_dict())
            
            self.db.store_content_analysis(screenshot_id, content_analysis.to_dict())
            
            return {
                "path": path,
                "screenshot_id": screenshot_id,
                "success": True
            }
            
        except Exception as e:
            return {
                "path": path,
                "error": str(e),
                "success": False
            }

# Usage
async def batch_processing_example():
    processor = BatchProcessor(max_workers=4)
    
    # Example screenshot paths
    screenshot_paths = [
        "screenshots/screenshot_1.png",
        "screenshots/screenshot_2.png",
        "screenshots/screenshot_3.png",
        # Add more paths as needed
    ]
    
    results = await processor.process_screenshot_batch(screenshot_paths)

asyncio.run(batch_processing_example())
```

### Optimization 2: Memory-Efficient Processing

Handle large datasets without running out of memory:

```python
import asyncio
import gc
from eidolon.storage.metadata_db import MetadataDatabase
from eidolon.core.memory import MemorySystem

class MemoryEfficientProcessor:
    def __init__(self, batch_size: int = 50):
        self.batch_size = batch_size
        self.db = MetadataDatabase()
        self.memory = MemorySystem()
        
    async def process_all_screenshots(self):
        """Process all screenshots in memory-efficient batches."""
        total_processed = 0
        batch_num = 0
        
        while True:
            # Get batch of unprocessed screenshots
            screenshots = self.db.get_unprocessed_screenshots(
                limit=self.batch_size,
                offset=batch_num * self.batch_size
            )
            
            if not screenshots:
                break
            
            print(f"🔄 Processing batch {batch_num + 1} ({len(screenshots)} items)")
            
            # Process batch
            await self._process_batch(screenshots)
            
            # Update counters
            total_processed += len(screenshots)
            batch_num += 1
            
            # Force garbage collection to free memory
            gc.collect()
            
            print(f"✅ Batch {batch_num} complete. Total processed: {total_processed}")
        
        print(f"🎉 All done! Processed {total_processed} screenshots")
    
    async def _process_batch(self, screenshots: list):
        """Process a single batch of screenshots."""
        for screenshot in screenshots:
            try:
                # Process screenshot
                await self._process_screenshot(screenshot)
                
                # Mark as processed
                self.db.mark_screenshot_processed(screenshot['id'])
                
            except Exception as e:
                print(f"❌ Failed to process {screenshot['id']}: {e}")
                # Mark as failed but continue
                self.db.mark_screenshot_failed(screenshot['id'], str(e))
    
    async def _process_screenshot(self, screenshot: dict):
        """Process individual screenshot with memory cleanup."""
        # Implementation here - keep it memory efficient
        pass

# Usage
async def memory_efficient_processing():
    processor = MemoryEfficientProcessor(batch_size=25)
    await processor.process_all_screenshots()

asyncio.run(memory_efficient_processing())
```

## Best Practices

### 1. Configuration Management

Organize configurations for different environments:

```python
# config/environments/development.yaml
observer:
  capture_interval: 5
  storage_path: "./dev_data"
  
analysis:
  routing:
    local_first: true
    cost_limit_daily: 1.0

logging:
  level: "DEBUG"

# config/environments/production.yaml
observer:
  capture_interval: 30
  storage_path: "/var/lib/eidolon/data"
  
analysis:
  routing:
    local_first: false
    cost_limit_daily: 10.0

logging:
  level: "INFO"
```

### 2. Error Handling and Logging

Implement comprehensive error handling:

```python
import logging
from eidolon.utils.logging import get_component_logger

class RobustObserver:
    def __init__(self):
        self.logger = get_component_logger("robust_observer")
        
    async def safe_capture_and_process(self):
        """Capture and process with comprehensive error handling."""
        try:
            # Capture screenshot
            screenshot = await self._safe_capture()
            if not screenshot:
                return None
            
            # Process with retries
            result = await self._process_with_retry(screenshot)
            return result
            
        except Exception as e:
            self.logger.error(f"Unexpected error in capture/process: {e}", exc_info=True)
            return None
    
    async def _safe_capture(self, max_retries: int = 3):
        """Capture screenshot with retries."""
        for attempt in range(max_retries):
            try:
                # Capture logic here
                return "mock_screenshot"
                
            except Exception as e:
                self.logger.warning(f"Capture attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    self.logger.error("All capture attempts failed")
                    return None
                
                # Wait before retry
                await asyncio.sleep(2 ** attempt)
    
    async def _process_with_retry(self, screenshot, max_retries: int = 2):
        """Process screenshot with retries."""
        for attempt in range(max_retries):
            try:
                # Processing logic here
                return "mock_result"
                
            except Exception as e:
                self.logger.warning(f"Processing attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    self.logger.error("All processing attempts failed")
                    raise
                
                await asyncio.sleep(1)
```

### 3. Resource Management

Implement proper resource cleanup:

```python
import asyncio
from contextlib import asynccontextmanager
from eidolon.core.observer import Observer

@asynccontextmanager
async def managed_observer():
    """Context manager for safe observer lifecycle."""
    observer = Observer()
    try:
        observer.start_monitoring()
        yield observer
    finally:
        observer.stop_monitoring()
        # Additional cleanup if needed

# Usage
async def safe_monitoring():
    async with managed_observer() as observer:
        # Use observer safely
        status = observer.get_status()
        print(f"Monitoring active: {status['running']}")
        
        # Wait for some captures
        await asyncio.sleep(30)
    
    # Observer is automatically stopped

asyncio.run(safe_monitoring())
```

### 4. Testing Strategies

Write comprehensive tests for custom components:

```python
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

class TestCustomAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return CustomAnalyzer()
    
    @pytest.fixture
    def sample_text(self):
        return "git commit -m 'fix bug' && python test.py"
    
    def test_extract_development_context(self, analyzer, sample_text):
        """Test development context extraction."""
        context = analyzer.extract_development_context(sample_text)
        
        assert 'git commit' in context['commands']
        assert len(context['commands']) == 1
    
    @patch('eidolon.core.analyzer.Analyzer.analyze_content')
    def test_analyze_content_enhancement(self, mock_analyze, analyzer):
        """Test enhanced content analysis."""
        # Setup mock
        mock_analyze.return_value = Mock(
            content_type="code",
            confidence=0.9,
            tags=["programming"],
            description="Code editor",
            vision_analysis=None
        )
        
        # Test
        result = analyzer.analyze_content("test.png", "git add .")
        
        # Verify enhancement
        assert "git" in result.tags
        assert "version-control" in result.tags
    
    @pytest.mark.asyncio
    async def test_async_processing(self):
        """Test asynchronous processing."""
        # Test async functionality
        result = await some_async_function()
        assert result is not None
```

This comprehensive guide provides practical examples and patterns for using Eidolon effectively in various scenarios. Each example builds on the core concepts while demonstrating real-world applications and best practices.