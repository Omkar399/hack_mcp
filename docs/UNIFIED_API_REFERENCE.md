# Eidolon AI Personal Assistant - Complete API Reference

This unified API reference covers all components across Phases 1-4 of the Eidolon system.

## Table of Contents

1. [Core Components](#core-components)
   - [Observer API](#observer-api)
   - [Analyzer API](#analyzer-api)
   - [Memory System API](#memory-system-api)
   - [Interface API](#interface-api)
2. [Storage APIs](#storage-apis)
   - [Vector Database API](#vector-database-api)
   - [Metadata Database API](#metadata-database-api)
   - [File Manager API](#file-manager-api)
3. [AI Model APIs](#ai-model-apis)
   - [Local Vision API](#local-vision-api)
   - [Cloud API Manager](#cloud-api-manager)
   - [Decision Engine API](#decision-engine-api)
4. [Utility APIs](#utility-apis)
   - [Configuration API](#configuration-api)
   - [Logging API](#logging-api)
   - [Monitoring API](#monitoring-api)
5. [CLI Commands](#cli-commands)

---

# Core Components

## Observer API

The Observer handles screenshot capture, change detection, and system monitoring.

### Class: `Observer`

```python
from eidolon.core.observer import Observer

class Observer:
    def __init__(self, config_override: Optional[Dict[str, Any]] = None):
        """
        Initialize the Observer.
        
        Args:
            config_override: Optional configuration overrides
        """
    
    def start_monitoring(self) -> None:
        """Start the monitoring thread for continuous capture."""
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring thread gracefully."""
    
    def capture_screenshot(self) -> Screenshot:
        """
        Capture a single screenshot.
        
        Returns:
            Screenshot object containing image data and metadata
            
        Raises:
            ScreenshotException: If capture fails
        """
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current observer status.
        
        Returns:
            Dictionary containing:
            - running: bool
            - capture_count: int
            - start_time: datetime
            - last_capture: datetime
            - performance_metrics: dict
        """
    
    def detect_changes(
        self, 
        prev_screenshot: Screenshot, 
        curr_screenshot: Screenshot
    ) -> bool:
        """
        Detect if significant changes occurred between screenshots.
        
        Returns:
            True if changes exceed threshold
        """
```

### Data Models

```python
@dataclass
class Screenshot:
    image: Image.Image
    timestamp: datetime
    hash: str
    file_path: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceMetrics:
    cpu_usage_percent: float
    memory_usage_mb: float
    disk_usage_percent: float
    capture_duration_ms: float
```

### Usage Examples

```python
# Basic monitoring
observer = Observer()
observer.start_monitoring()
# ... let it run ...
observer.stop_monitoring()

# Manual capture
screenshot = observer.capture_screenshot()
print(f"Captured at: {screenshot.timestamp}")

# Custom configuration
observer = Observer({
    "capture_interval": 30,
    "activity_threshold": 0.1
})

# Check status
status = observer.get_status()
print(f"Captures: {status['capture_count']}")
```

---

## Analyzer API

The Analyzer performs OCR, content classification, and AI-powered analysis.

### Class: `Analyzer`

```python
from eidolon.core.analyzer import Analyzer

class Analyzer:
    def __init__(self, config: Optional[Config] = None):
        """Initialize analyzer with OCR and AI capabilities."""
    
    def extract_text(self, image_path: Union[str, Path]) -> OCRResult:
        """
        Extract text from image using OCR.
        
        Args:
            image_path: Path to image file
            
        Returns:
            OCRResult with extracted text and confidence
        """
    
    def analyze_content(
        self, 
        image_path: Union[str, Path],
        include_vision: bool = True
    ) -> ContentAnalysis:
        """
        Analyze image content with AI.
        
        Returns:
            ContentAnalysis with classification and description
        """
    
    def analyze_with_florence(self, image_path: Union[str, Path]) -> VisionAnalysis:
        """
        Use Florence-2 model for advanced vision analysis.
        
        Returns:
            VisionAnalysis with scene understanding
        """
```

### Data Models

```python
@dataclass
class OCRResult:
    text: str
    confidence: float
    word_count: int
    language: str
    regions: List[TextRegion]
    metadata: Dict[str, Any]

@dataclass
class ContentAnalysis:
    content_type: str  # 'document', 'browser', 'terminal', etc.
    description: str
    confidence: float
    tags: List[str]
    ui_elements: List[UIElement]
    vision_analysis: Optional[VisionAnalysis] = None

@dataclass
class VisionAnalysis:
    scene_type: str
    description: str
    objects_detected: List[str]
    confidence: float
    model_used: str
```

### Usage Examples

```python
analyzer = Analyzer()

# OCR extraction
ocr_result = analyzer.extract_text("screenshot.png")
print(f"Text: {ocr_result.text[:100]}...")
print(f"Confidence: {ocr_result.confidence:.2f}")

# Content analysis
analysis = analyzer.analyze_content("screenshot.png")
print(f"Type: {analysis.content_type}")
print(f"Tags: {', '.join(analysis.tags)}")

# Vision analysis
if analyzer._florence_available:
    vision = analyzer.analyze_with_florence("screenshot.png")
    print(f"Scene: {vision.scene_type}")
    print(f"Objects: {', '.join(vision.objects_detected)}")
```

---

## Memory System API

Enhanced memory system with NLP, semantic search, and RAG capabilities.

### Class: `MemorySystem`

```python
from eidolon.core.memory import MemorySystem

class MemorySystem:
    def __init__(self, config: Optional[Config] = None):
        """Initialize memory system with all components."""
    
    async def process_natural_language_query(self, query: str) -> MemoryResponse:
        """
        Process natural language query with intent recognition.
        
        Args:
            query: Natural language query
            
        Returns:
            MemoryResponse with results and optional RAG response
        """
    
    def parse_query_intent(self, query: str) -> QueryIntent:
        """
        Parse user intent from natural language.
        
        Returns:
            QueryIntent with type, confidence, and parameters
        """
    
    def parse_time_expressions(self, query: str) -> Dict[str, datetime]:
        """
        Extract time-based filters from query.
        
        Returns:
            Dictionary with 'start' and 'end' datetime objects
        """
    
    async def generate_rag_response(
        self,
        query: str,
        search_results: List[SearchResult]
    ) -> Optional[str]:
        """
        Generate contextual response using RAG.
        
        Returns:
            Generated response or None if unavailable
        """
    
    def store_capture(
        self,
        screenshot: Screenshot,
        ocr_result: Optional[OCRResult] = None,
        content_analysis: Optional[ContentAnalysis] = None
    ) -> bool:
        """Store capture data in all databases."""
    
    def search(
        self,
        query: str,
        search_type: str = "hybrid",
        limit: int = 10,
        **filters
    ) -> List[SearchResult]:
        """
        Search memory with various methods.
        
        Args:
            query: Search query
            search_type: 'text', 'semantic', 'hybrid'
            limit: Maximum results
            **filters: Additional filters
        """
```

### Data Models

```python
@dataclass
class MemoryResponse:
    query: str
    intent: str
    results: List[SearchResult]
    response: Optional[str]
    confidence: float
    processing_time: float

@dataclass
class QueryIntent:
    type: str  # 'search', 'summarize', 'analyze', 'compare', 'timeline'
    confidence: float
    search_terms: List[str]
    filters: Dict[str, Any]
    time_range: Optional[Dict[str, datetime]]

@dataclass
class SearchResult:
    id: str
    content: str
    relevance: float
    timestamp: datetime
    metadata: Dict[str, Any]
    source: str  # 'text_search', 'semantic_search', 'hybrid'
```

### Usage Examples

```python
memory = MemorySystem()

# Natural language query
response = await memory.process_natural_language_query(
    "Find all Python code from yesterday"
)
print(f"Intent: {response.intent}")
print(f"Found: {len(response.results)} results")

# Parse intent
intent = memory.parse_query_intent("Summarize my work today")
print(f"Type: {intent.type}")
print(f"Confidence: {intent.confidence}")

# Time parsing
time_range = memory.parse_time_expressions("from last Monday to Friday")
print(f"Start: {time_range['start']}")
print(f"End: {time_range['end']}")

# Direct search
results = memory.search(
    "machine learning",
    search_type="semantic",
    limit=5
)
```

---

# Storage APIs

## Vector Database API

ChromaDB-based vector storage for semantic search.

### Class: `VectorDatabase`

```python
from eidolon.storage.vector_db import VectorDatabase

class VectorDatabase:
    def __init__(self, db_path: Optional[str] = None):
        """Initialize ChromaDB vector database."""
    
    def store_content(
        self,
        screenshot_id: str,
        content_analysis: Dict[str, Any],
        extracted_text: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store content with vector embedding."""
    
    def semantic_search(
        self,
        query: str,
        n_results: int = 10,
        content_type_filter: Optional[str] = None,
        min_confidence: float = 0.0,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """Perform semantic similarity search."""
    
    def hybrid_search(
        self,
        query: str,
        n_results: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        **filters
    ) -> List[Dict[str, Any]]:
        """Combine semantic and keyword search."""
    
    def get_similar_content(
        self,
        screenshot_id: str,
        n_results: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Find similar content to a specific screenshot."""
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
```

### Class: `EmbeddingGenerator`

```python
class EmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize embedding model."""
    
    def generate_text_embedding(self, text: str) -> Optional[List[float]]:
        """Generate 384-dimensional embedding for text."""
    
    def generate_content_embedding(
        self,
        content_analysis: Dict[str, Any]
    ) -> Optional[List[float]]:
        """Generate embedding for content analysis."""
```

---

## Metadata Database API

SQLite database with full-text search.

### Class: `MetadataDatabase`

```python
from eidolon.storage.metadata_db import MetadataDatabase

class MetadataDatabase:
    def __init__(self, db_path: Optional[str] = None):
        """Initialize SQLite database with FTS5."""
    
    def store_screenshot(
        self,
        file_path: str,
        timestamp: datetime,
        hash: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Store screenshot metadata."""
    
    def store_ocr_result(
        self,
        screenshot_id: int,
        text: str,
        confidence: float,
        language: str = "en",
        word_count: int = 0
    ) -> int:
        """Store OCR extraction results."""
    
    def store_content_analysis(
        self,
        screenshot_id: int,
        content_type: str,
        description: str,
        confidence: float,
        tags: Optional[List[str]] = None,
        vision_analysis: Optional[Dict[str, Any]] = None
    ) -> int:
        """Store content analysis results."""
    
    def search_text(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Full-text search using FTS5."""
    
    def search_by_content_type(
        self,
        content_type: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search by content type."""
    
    def search_by_time_range(
        self,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search within time range."""
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
```

---

# AI Model APIs

## Cloud API Manager

Unified interface for cloud AI providers.

### Class: `CloudAPIManager`

```python
from eidolon.models.cloud_api import CloudAPIManager

class CloudAPIManager:
    def __init__(self, config: Optional[Config] = None):
        """Initialize with available API providers."""
    
    async def analyze_image(
        self,
        image_path: Union[str, Path],
        prompt: str,
        preferred_provider: Optional[str] = None,
        fallback: bool = True
    ) -> Optional[CloudAPIResponse]:
        """
        Analyze image using cloud AI.
        
        Args:
            image_path: Path to image
            prompt: Analysis prompt
            preferred_provider: 'gemini', 'claude', or 'openai'
            fallback: Try other providers if preferred fails
        """
    
    async def analyze_text(
        self,
        text: str,
        analysis_type: str = "general",
        preferred_provider: Optional[str] = None
    ) -> Optional[CloudAPIResponse]:
        """Analyze text using cloud AI."""
    
    def get_available_providers(self) -> List[str]:
        """Get list of providers with API keys."""
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics and costs."""
```

### Individual API Classes

```python
class GeminiAPI:
    """Google Gemini API (gemini-1.5-flash)"""
    
class ClaudeAPI:
    """Anthropic Claude API (claude-3-5-sonnet-20241022)"""
    
class OpenAIAPI:
    """OpenAI GPT API (gpt-4o)"""
```

### Data Models

```python
@dataclass
class CloudAPIResponse:
    provider: str
    content: str
    confidence: float
    cost: float
    duration: float
    metadata: Dict[str, Any]
```

---

## Decision Engine API

Intelligent routing between local and cloud AI.

### Class: `DecisionEngine`

```python
from eidolon.models.decision_engine import DecisionEngine

class DecisionEngine:
    def __init__(self, config: Optional[Config] = None):
        """Initialize decision engine with routing rules."""
    
    def make_routing_decision(
        self,
        request: AnalysisRequest,
        available_providers: List[str]
    ) -> RoutingDecision:
        """Decide whether to use local or cloud AI."""
    
    def estimate_cost(
        self,
        request: AnalysisRequest,
        provider: str
    ) -> float:
        """Estimate cost for cloud API usage."""
    
    def get_daily_usage(self) -> Dict[str, Any]:
        """Get current daily usage statistics."""
    
    def update_usage(
        self,
        provider: str,
        cost: float,
        tokens: int
    ) -> None:
        """Update usage tracking."""
```

### Data Models

```python
@dataclass
class AnalysisRequest:
    content_type: str
    text_length: int
    has_image: bool
    importance: float
    quality_requirement: float
    user_query: Optional[str] = None

@dataclass
class RoutingDecision:
    use_cloud: bool
    provider: Optional[str]
    reasoning: str
    estimated_cost: float
    estimated_quality: float
    confidence: float
```

---

# Utility APIs

## Configuration API

### Function: `get_config`

```python
from eidolon.utils.config import get_config

def get_config() -> Config:
    """
    Load configuration from file and environment.
    
    Returns:
        Config object with all settings
    """

# Usage
config = get_config()
print(f"Capture interval: {config.observer.capture_interval}")
print(f"AI model: {config.analysis.local_models.vision}")
```

### Class: `Config`

```python
@dataclass
class Config:
    observer: ObserverConfig
    analysis: AnalysisConfig
    memory: MemoryConfig
    privacy: PrivacyConfig
    performance: PerformanceConfig
```

---

## Logging API

### Function: `get_component_logger`

```python
from eidolon.utils.logging import get_component_logger

def get_component_logger(component_name: str) -> logging.Logger:
    """Get logger for specific component."""

# Usage
logger = get_component_logger("observer")
logger.info("Starting capture")
logger.error("Capture failed", exc_info=True)
```

### Decorators

```python
from eidolon.utils.logging import log_performance, log_exceptions

@log_performance
def slow_function():
    """This function's performance will be logged."""
    pass

@log_exceptions("my_component")
def risky_function():
    """Exceptions will be logged with component context."""
    pass
```

---

## Monitoring API

### Class: `PerformanceMonitor`

```python
from eidolon.utils.monitoring import PerformanceMonitor

class PerformanceMonitor:
    def get_metrics(self) -> PerformanceMetrics:
        """Get current system metrics."""
    
    def check_resources(self) -> ResourceStatus:
        """Check if resources are within limits."""
    
    @contextmanager
    def track_operation(self, operation_name: str):
        """Track performance of an operation."""
```

---

# CLI Commands

## Basic Commands

```bash
# Show help
python -m eidolon --help

# Show version
python -m eidolon version

# Check status
python -m eidolon status [--json]
```

## Capture Commands

```bash
# Start capture
python -m eidolon capture [options]
  --interval SECONDS     # Capture interval (default: 10)
  --background          # Run in background
  --storage PATH        # Custom storage path
  
# Examples
python -m eidolon capture --interval 30
python -m eidolon capture --background
```

## Search Commands

```bash
# Search captures
python -m eidolon search QUERY [options]
  --limit N             # Result limit (default: 10)
  --format FORMAT       # Output format: text, json
  --content-type TYPE   # Filter by type
  --from DATE          # Start date
  --to DATE            # End date
  --semantic           # Use semantic search
  
# Examples
python -m eidolon search "python code" --limit 5
python -m eidolon search "error" --content-type terminal
python -m eidolon search "meeting notes" --from "2024-01-01"
```

## Natural Language Queries

```bash
# Phase 4 natural language support
python -m eidolon search "What did I work on yesterday?"
python -m eidolon search "Summarize my Python coding today"
python -m eidolon search "Find all terminal errors from this week"
```

## Management Commands

```bash
# Clean old data
python -m eidolon cleanup --days N

# Export data
python -m eidolon export --path OUTPUT_PATH [--format json|csv]

# Show statistics
python -m eidolon stats

# Validate setup
python -m eidolon validate
```

---

## Error Handling

### Common Exceptions

```python
# Observer errors
class ScreenshotException(Exception):
    """Failed to capture screenshot."""

class PermissionException(Exception):
    """Missing required permissions."""

# Storage errors  
class DatabaseException(Exception):
    """Database operation failed."""

class VectorDatabaseException(Exception):
    """Vector database operation failed."""

# AI errors
class ModelLoadException(Exception):
    """Failed to load AI model."""

class CloudAPIException(Exception):
    """Cloud API request failed."""

# Memory errors
class SearchException(Exception):
    """Search operation failed."""

class RAGException(Exception):
    """RAG generation failed."""
```

### Error Handling Pattern

```python
try:
    result = await api_manager.analyze_image(image_path, prompt)
except CloudAPIException as e:
    logger.error(f"Cloud analysis failed: {e}")
    # Fallback to local
    result = analyzer.analyze_content(image_path)
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    raise
```

---

## Performance Considerations

### Async Operations

```python
# Good: Non-blocking async
async def process_multiple(items):
    tasks = [process_item(item) for item in items]
    results = await asyncio.gather(*tasks)
    return results

# Bad: Sequential processing
def process_multiple_sync(items):
    results = []
    for item in items:
        results.append(process_item_sync(item))
    return results
```

### Batch Operations

```python
# Good: Batch database operations
def store_multiple_screenshots(screenshots):
    with db.begin_transaction():
        for screenshot in screenshots:
            db.store_screenshot(screenshot)

# Bad: Individual transactions
def store_screenshots_slow(screenshots):
    for screenshot in screenshots:
        db.store_screenshot(screenshot)  # Each has own transaction
```

### Caching

```python
# Built-in caching for embeddings
@lru_cache(maxsize=1000)
def get_cached_embedding(text_hash: str) -> List[float]:
    return embedding_generator.generate_text_embedding(text)
```

---

*Last updated: 2025-07-19 | Version: 0.1.0 | API Reference for Phases 1-4*