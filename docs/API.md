# Eidolon AI Personal Assistant - API Documentation

## Table of Contents

1. [Overview](#overview)
2. [Core Components](#core-components)
3. [Observer API](#observer-api)
4. [Analyzer API](#analyzer-api)
5. [Memory System API](#memory-system-api)
6. [Storage APIs](#storage-apis)
7. [Cloud AI APIs](#cloud-ai-apis)
8. [Configuration System](#configuration-system)
9. [CLI Interface](#cli-interface)
10. [Error Handling](#error-handling)
11. [Performance Monitoring](#performance-monitoring)

## Overview

Eidolon AI Personal Assistant provides a comprehensive Python API for building AI-powered personal assistant systems with screenshot monitoring, content analysis, and semantic search capabilities.

### Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Observer      │    │    Analyzer     │    │  Memory System  │
│  (Screenshots)  │───▶│  (AI Analysis)  │───▶│ (Vector Search) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Storage Layer   │    │   Cloud APIs    │    │ Interface Layer │
│ (DB/Files)      │    │ (Gemini/Claude) │    │   (CLI/API)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Core Components

### Observer (Screenshot Monitoring)

The Observer component handles automated screenshot capture with intelligent change detection.

```python
from eidolon.core.observer import Observer

# Initialize observer
observer = Observer()

# Start monitoring
observer.start_monitoring()

# Get status
status = observer.get_status()
print(f"Running: {status['running']}")
print(f"Captures: {status['capture_count']}")

# Stop monitoring
observer.stop_monitoring()
```

### Analyzer (AI-Powered Analysis)

The Analyzer component extracts text and analyzes content using OCR and AI models.

```python
from eidolon.core.analyzer import Analyzer

# Initialize analyzer
analyzer = Analyzer()

# Extract text from screenshot
extracted_text = analyzer.extract_text("/path/to/screenshot.png")
print(f"Text: {extracted_text.text}")
print(f"Confidence: {extracted_text.confidence}")

# Analyze content
content_analysis = analyzer.analyze_content("/path/to/screenshot.png", extracted_text.text)
print(f"Type: {content_analysis.content_type}")
print(f"Tags: {content_analysis.tags}")
```

### Memory System (Semantic Search)

The Memory component provides semantic search and knowledge management.

```python
from eidolon.core.memory import MemorySystem

# Initialize memory system
memory = MemorySystem()

# Store content
await memory.store_content(
    screenshot_id="screenshot_123",
    content_analysis={"description": "Code editor with Python"},
    extracted_text="def hello_world(): print('Hello')"
)

# Search content
results = await memory.semantic_search("Python code")
for result in results:
    print(f"Found: {result.content}")
```

## Observer API

### Class: `Observer`

Handles automated screenshot capture and monitoring.

#### Methods

##### `__init__(config_override: Optional[Dict] = None)`

Initialize the Observer with optional configuration override.

**Parameters:**
- `config_override` (Dict, optional): Override specific configuration values

**Example:**
```python
observer = Observer(config_override={
    "capture_interval": 5,  # Capture every 5 seconds
    "storage_path": "./my_screenshots"
})
```

##### `start_monitoring() -> None`

Start the background monitoring thread.

**Raises:**
- `RuntimeError`: If monitoring is already running

##### `stop_monitoring() -> None`

Stop the monitoring thread and wait for completion.

##### `capture_screenshot() -> Screenshot`

Capture a single screenshot immediately.

**Returns:**
- `Screenshot`: Screenshot object with metadata

**Example:**
```python
screenshot = observer.capture_screenshot()
print(f"Size: {screenshot.width}x{screenshot.height}")
print(f"Timestamp: {screenshot.timestamp}")
```

##### `get_status() -> Dict[str, Any]`

Get current monitoring status and performance metrics.

**Returns:**
- `Dict`: Status information including running state, capture count, and metrics

##### `cleanup_old_screenshots(days_to_keep: int = None) -> int`

Remove old screenshots to manage storage space.

**Parameters:**
- `days_to_keep` (int, optional): Number of days to retain. Uses config default if None.

**Returns:**
- `int`: Number of files deleted

### Class: `Screenshot`

Represents a captured screenshot with metadata.

#### Attributes

- `timestamp` (datetime): When the screenshot was taken
- `width` (int): Screenshot width in pixels
- `height` (int): Screenshot height in pixels
- `file_path` (str): Path to saved screenshot file
- `monitor_info` (Dict): Information about monitor setup
- `window_info` (Dict): Information about active window
- `performance_metrics` (Dict): Capture performance data

#### Methods

##### `to_dict() -> Dict[str, Any]`

Convert screenshot to dictionary representation.

##### `save(file_path: str) -> None`

Save screenshot to specified path.

## Analyzer API

### Class: `Analyzer`

Performs OCR text extraction and AI-powered content analysis.

#### Methods

##### `__init__()`

Initialize the Analyzer with default configuration.

##### `extract_text(image_path: Union[str, Path]) -> ExtractedText`

Extract text from an image using OCR.

**Parameters:**
- `image_path`: Path to image file

**Returns:**
- `ExtractedText`: Text extraction results with confidence scores

**Example:**
```python
analyzer = Analyzer()
result = analyzer.extract_text("screenshot.png")

print(f"Text: {result.text}")
print(f"Word count: {result.word_count}")
print(f"Confidence: {result.confidence}")

# Access individual text regions
for region in result.regions:
    print(f"Region: {region.text} (confidence: {region.confidence})")
```

##### `analyze_content(image_path: Union[str, Path], extracted_text: str = "") -> ContentAnalysis`

Analyze screenshot content and classify it.

**Parameters:**
- `image_path`: Path to image file
- `extracted_text`: Previously extracted text (optional)

**Returns:**
- `ContentAnalysis`: Analysis results with content type, tags, and description

**Example:**
```python
analysis = analyzer.analyze_content("screenshot.png", "print('hello')")

print(f"Content type: {analysis.content_type}")  # e.g., "code", "browser", "document"
print(f"Tags: {analysis.tags}")  # e.g., ["python", "programming"]
print(f"Description: {analysis.description}")
print(f"Confidence: {analysis.confidence}")
```

##### `classify_content_type(text: str) -> str`

Classify content type based on extracted text.

**Parameters:**
- `text`: Text content to classify

**Returns:**
- `str`: Content type ("code", "browser", "document", "general")

### Class: `ExtractedText`

Represents OCR text extraction results.

#### Attributes

- `text` (str): Extracted text content
- `confidence` (float): Overall extraction confidence (0-100)
- `language` (str): Detected language
- `word_count` (int): Number of words extracted
- `regions` (List[TextRegion]): Individual text regions

### Class: `ContentAnalysis`

Represents content analysis results.

#### Attributes

- `content_type` (str): Classified content type
- `confidence` (float): Analysis confidence (0-1)
- `tags` (List[str]): Content tags
- `description` (str): Generated description
- `vision_analysis` (Optional[VisionAnalysis]): AI vision analysis results

## Memory System API

### Class: `MemorySystem`

Provides semantic search and knowledge management capabilities.

#### Methods

##### `__init__()`

Initialize the Memory System with vector database.

##### `async store_content(screenshot_id: str, content_analysis: Dict[str, Any], extracted_text: str = "") -> bool`

Store content in the memory system with semantic embeddings.

**Parameters:**
- `screenshot_id`: Unique identifier for the screenshot
- `content_analysis`: Analysis results dictionary
- `extracted_text`: OCR extracted text

**Returns:**
- `bool`: True if stored successfully

##### `async semantic_search(query: str, n_results: int = 10, content_type_filter: str = None) -> List[SearchResult]`

Search stored content using semantic similarity.

**Parameters:**
- `query`: Search query text
- `n_results`: Maximum number of results
- `content_type_filter`: Filter by content type

**Returns:**
- `List[SearchResult]`: Ranked search results

**Example:**
```python
results = await memory.semantic_search("Python programming", n_results=5)

for result in results:
    print(f"Similarity: {result.similarity:.2f}")
    print(f"Content: {result.content}")
    print(f"Metadata: {result.metadata}")
```

##### `async generate_rag_response(query: str, context_limit: int = 5) -> MemoryResponse`

Generate AI response using Retrieval-Augmented Generation.

**Parameters:**
- `query`: User query
- `context_limit`: Number of context documents to retrieve

**Returns:**
- `MemoryResponse`: AI-generated response with sources

### Class: `SearchResult`

Represents a search result from the memory system.

#### Attributes

- `content` (str): Content text
- `similarity` (float): Similarity score (0-1)
- `metadata` (Dict): Associated metadata
- `timestamp` (datetime): When content was stored

## Storage APIs

### MetadataDatabase

SQLite-based storage for screenshot metadata and analysis results.

```python
from eidolon.storage.metadata_db import MetadataDatabase

db = MetadataDatabase()

# Store screenshot metadata
screenshot_id = db.store_screenshot({
    "timestamp": "2023-12-01T10:00:00",
    "width": 1920,
    "height": 1080,
    "file_path": "/path/to/screenshot.png"
})

# Store OCR results
db.store_ocr_result(screenshot_id, {
    "text": "Hello world",
    "confidence": 95.5,
    "word_count": 2
})

# Search by text
results = db.search_text("Hello")
for result in results:
    print(f"Found in screenshot: {result['timestamp']}")
```

### VectorDatabase

ChromaDB-based vector storage for semantic search.

```python
from eidolon.storage.vector_db import VectorDatabase

vector_db = VectorDatabase()

# Store content with embedding
success = vector_db.store_content(
    screenshot_id="123",
    content_analysis={"description": "Code editor"},
    extracted_text="def hello(): pass"
)

# Semantic search
results = vector_db.semantic_search("programming code", n_results=5)
for result in results:
    print(f"Match: {result['document']} (similarity: {result['similarity']})")
```

## Cloud AI APIs

### Gemini API

```python
from eidolon.models.cloud_api import GeminiAPI

# Initialize with API key
gemini = GeminiAPI(api_key="your-api-key")

# Analyze image
response = await gemini.analyze_image(
    image_path="screenshot.png",
    prompt="Describe what you see in this image"
)

print(f"Description: {response.content}")
print(f"Model: {response.model_used}")
```

### Claude API

```python
from eidolon.models.cloud_api import ClaudeAPI

claude = ClaudeAPI(api_key="your-api-key")

# Generate text response
response = await claude.generate_response(
    prompt="Explain this code",
    context="def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
)
```

### Cloud API Manager

```python
from eidolon.models.cloud_api import CloudAPIManager

# Automatic provider selection and fallback
manager = CloudAPIManager()

# Analyze with best available provider
response = await manager.analyze_with_best_available(
    content_type="code",
    image_path="screenshot.png",
    prompt="Analyze this code"
)
```

## Configuration System

### Loading Configuration

```python
from eidolon.utils.config import get_config, load_config

# Load default configuration
config = get_config()

# Load from specific file
config = load_config("custom_config.yaml")

# Access configuration values
print(f"Capture interval: {config.observer.capture_interval}")
print(f"Storage path: {config.observer.storage_path}")
```

### Configuration Structure

```yaml
observer:
  capture_interval: 10        # Seconds between captures
  storage_path: "./data"      # Storage directory
  max_storage_gb: 50         # Storage limit
  
analysis:
  local_models:
    vision: "microsoft/florence-2-base"
  cloud_apis:
    gemini_key: "${GEMINI_API_KEY}"
    
memory:
  vector_db: "chromadb"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  
privacy:
  local_only_mode: false
  auto_redaction: true
```

## CLI Interface

### Basic Commands

```bash
# Start monitoring
python -m eidolon capture

# Search content
python -m eidolon search "python code"

# Check status
python -m eidolon status

# Clean up old data
python -m eidolon cleanup --days 30

# Export data
python -m eidolon export --path backup.json
```

### Advanced Usage

```bash
# Custom capture interval
python -m eidolon capture --interval 5

# Background mode
python -m eidolon capture --background

# Filtered search
python -m eidolon search "code" --content-type development --limit 20

# JSON output
python -m eidolon status --json
```

## Error Handling

### Exception Types

- `ConfigurationError`: Configuration file issues
- `StorageError`: Database or file system errors
- `AnalysisError`: AI model or OCR failures
- `NetworkError`: Cloud API connection issues

### Example Error Handling

```python
from eidolon.core.observer import Observer
from eidolon.utils.exceptions import StorageError, ConfigurationError

try:
    observer = Observer()
    observer.start_monitoring()
except ConfigurationError as e:
    print(f"Configuration problem: {e}")
except StorageError as e:
    print(f"Storage issue: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Monitoring

### Resource Monitoring

```python
# Check system resource usage
observer = Observer()
status = observer.get_status()

metrics = status['performance_metrics']
print(f"Memory usage: {metrics['memory_usage_mb']} MB")
print(f"CPU usage: {metrics['cpu_usage_percent']}%")
print(f"Captures per minute: {metrics['captures_per_minute']}")
```

### Optimization Tips

1. **Storage Management**: Regularly clean up old screenshots
2. **Resource Limits**: Configure appropriate CPU and memory limits
3. **Capture Interval**: Balance between data capture and performance
4. **Local vs Cloud**: Use local models for faster analysis when possible

### Performance Benchmarks

| Component | Typical Performance |
|-----------|-------------------|
| Screenshot Capture | <100ms |
| OCR Text Extraction | 500ms - 2s |
| Local AI Analysis | 1-5s |
| Cloud AI Analysis | 2-10s |
| Vector Search | <500ms |
| Database Operations | <100ms |

## Integration Examples

### Complete Workflow

```python
import asyncio
from eidolon.core.observer import Observer
from eidolon.core.analyzer import Analyzer  
from eidolon.core.memory import MemorySystem

async def main():
    # Initialize components
    observer = Observer()
    analyzer = Analyzer()
    memory = MemorySystem()
    
    # Capture and analyze
    screenshot = observer.capture_screenshot()
    extracted_text = analyzer.extract_text(screenshot.file_path)
    content_analysis = analyzer.analyze_content(screenshot.file_path, extracted_text.text)
    
    # Store in memory
    await memory.store_content(
        screenshot_id=screenshot.id,
        content_analysis=content_analysis.to_dict(),
        extracted_text=extracted_text.text
    )
    
    # Search for similar content
    results = await memory.semantic_search("code editor")
    print(f"Found {len(results)} similar items")

# Run the workflow
asyncio.run(main())
```

This API documentation provides comprehensive coverage of all major components and their usage patterns. For specific implementation details, refer to the source code and inline documentation.