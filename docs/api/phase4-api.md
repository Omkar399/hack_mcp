# Phase 4 API Documentation - Cloud AI & Semantic Memory

## Overview

Phase 4 introduces advanced AI capabilities including vector database integration, cloud AI APIs, natural language processing, and RAG (Retrieval-Augmented Generation) for contextual responses.

## Core Components

### 1. Vector Database (`eidolon.storage.vector_db`)

#### EmbeddingGenerator Class

Generates embeddings for text and content using Sentence Transformers.

```python
class EmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2")
    
    def generate_text_embedding(self, text: str) -> Optional[List[float]]
        """Generate 384-dimensional embedding for text content."""
    
    def generate_content_embedding(self, content_analysis: Dict[str, Any]) -> Optional[List[float]]
        """Generate embedding for comprehensive content analysis."""
```

#### VectorDatabase Class

ChromaDB-based vector database for semantic search and storage.

```python
class VectorDatabase:
    def __init__(self, db_path: Optional[str] = None)
    
    def store_content(
        self, 
        screenshot_id: str, 
        content_analysis: Dict[str, Any],
        extracted_text: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool
        """Store content with vector embedding."""
    
    def semantic_search(
        self,
        query: str,
        n_results: int = 10,
        content_type_filter: Optional[str] = None,
        min_confidence: float = 0.0,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]
        """Perform semantic similarity search."""
    
    def hybrid_search(
        self,
        query: str,
        n_results: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        **filters
    ) -> List[Dict[str, Any]]
        """Combine semantic and keyword search."""
    
    def get_similar_content(
        self,
        screenshot_id: str,
        n_results: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]
        """Find content similar to a specific screenshot."""
```

### 2. Cloud AI APIs (`eidolon.models.cloud_api`)

#### CloudAPIManager Class

Unified interface for managing multiple cloud AI providers.

```python
class CloudAPIManager:
    def __init__(self)
    
    async def analyze_image(
        self, 
        image_path: Union[str, Path], 
        prompt: str,
        preferred_provider: Optional[str] = None,
        fallback: bool = True
    ) -> Optional[CloudAPIResponse]
        """Analyze image using cloud AI with fallback support."""
    
    async def analyze_text(
        self, 
        text: str, 
        analysis_type: str = "general",
        preferred_provider: Optional[str] = None
    ) -> Optional[CloudAPIResponse]
        """Analyze text using cloud AI."""
    
    def get_available_providers(self) -> List[str]
        """Get list of available providers with API keys."""
    
    def get_usage_stats(self) -> Dict[str, Any]
        """Get usage statistics and costs."""
```

#### Individual API Classes

```python
class GeminiAPI:
    """Google Gemini API integration (gemini-1.5-flash)"""
    
class ClaudeAPI:
    """Anthropic Claude API integration (claude-3-5-sonnet-20241022)"""
    
class OpenAIAPI:
    """OpenAI GPT API integration (gpt-4o)"""
```

### 3. Decision Engine (`eidolon.models.decision_engine`)

#### DecisionEngine Class

Intelligent routing between local and cloud AI based on various factors.

```python
class DecisionEngine:
    def __init__(self, config: Optional[Config] = None)
    
    def make_routing_decision(
        self, 
        request: AnalysisRequest, 
        available_providers: List[str]
    ) -> RoutingDecision
        """Decide whether to use local or cloud AI."""
    
    def estimate_cost(self, request: AnalysisRequest, provider: str) -> float
        """Estimate cost for cloud API usage."""
    
    def get_daily_usage(self) -> Dict[str, Any]
        """Get current daily usage statistics."""
```

#### Data Models

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

### 4. Enhanced Memory System (`eidolon.core.memory`)

#### MemorySystem Class

Advanced memory system with NLP and RAG capabilities.

```python
class MemorySystem:
    def __init__(self, config: Optional[Config] = None)
    
    async def process_natural_language_query(self, query: str) -> MemoryResponse
        """Process natural language queries with intent recognition."""
    
    def parse_query_intent(self, query: str) -> QueryIntent
        """Parse user intent from natural language."""
    
    def parse_time_expressions(self, query: str) -> Dict[str, datetime]
        """Extract time-based filters from queries."""
    
    async def generate_rag_response(
        self, 
        query: str, 
        search_results: List[SearchResult]
    ) -> Optional[str]
        """Generate contextual response using RAG."""
```

## Usage Examples

### Basic Semantic Search

```python
from eidolon.storage.vector_db import VectorDatabase

# Initialize vector database
vector_db = VectorDatabase()

# Store content
vector_db.store_content(
    screenshot_id="screenshot_123",
    content_analysis={
        "content_type": "document",
        "description": "Python code for data analysis",
        "tags": ["python", "pandas", "data science"]
    },
    extracted_text="import pandas as pd\ndf = pd.read_csv('data.csv')"
)

# Semantic search
results = vector_db.semantic_search("python data manipulation")
for result in results:
    print(f"Similarity: {result['similarity']:.3f} - {result['document'][:50]}...")
```

### Cloud AI Analysis

```python
import asyncio
from eidolon.models.cloud_api import CloudAPIManager

async def analyze_with_cloud():
    api_manager = CloudAPIManager()
    
    # Check available providers
    providers = api_manager.get_available_providers()
    print(f"Available providers: {providers}")
    
    # Analyze image
    response = await api_manager.analyze_image(
        "screenshot.png",
        "Describe what the user is working on"
    )
    
    if response:
        print(f"Provider: {response.provider}")
        print(f"Response: {response.content}")
        print(f"Cost: ${response.cost:.4f}")

asyncio.run(analyze_with_cloud())
```

### Natural Language Query Processing

```python
import asyncio
from eidolon.core.memory import MemorySystem

async def process_query():
    memory = MemorySystem()
    
    # Process various query types
    queries = [
        "Find all Python code from yesterday",
        "Summarize my work on machine learning",
        "Show me terminal commands from this week",
        "What errors did I encounter today?"
    ]
    
    for query in queries:
        response = await memory.process_natural_language_query(query)
        print(f"\nQuery: {query}")
        print(f"Intent: {response.intent}")
        print(f"Results: {len(response.results)}")
        print(f"Response: {response.response[:100]}...")

asyncio.run(process_query())
```

### Intelligent Routing Decision

```python
from eidolon.models.decision_engine import DecisionEngine, AnalysisRequest

# Initialize decision engine
decision_engine = DecisionEngine()

# Create analysis request
request = AnalysisRequest(
    content_type="code",
    text_length=500,
    has_image=True,
    importance=0.8,
    quality_requirement=0.9
)

# Get routing decision
decision = decision_engine.make_routing_decision(
    request, 
    available_providers=["gemini", "claude"]
)

print(f"Use cloud: {decision.use_cloud}")
print(f"Provider: {decision.provider}")
print(f"Reasoning: {decision.reasoning}")
print(f"Estimated cost: ${decision.estimated_cost:.4f}")
```

## Configuration

### Vector Database Settings

```yaml
memory:
  vector_db: "chromadb"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  chunk_size: 512
  overlap: 50
  db_path: "./data/memory/metadata.db"
```

### Cloud AI Configuration

```yaml
analysis:
  cloud_apis:
    gemini_key: "${GEMINI_API_KEY}"
    claude_key: "${CLAUDE_API_KEY}"
    openai_key: "${OPENAI_API_KEY}"
  routing:
    importance_threshold: 0.7
    cost_limit_daily: 10.0
    local_first: true
```

## Performance Considerations

### Embedding Generation
- Model: all-MiniLM-L6-v2 (384 dimensions)
- Speed: ~100ms per text embedding
- Memory: ~500MB model size

### Vector Search Performance
- ChromaDB indexing: O(log n)
- Typical query time: <100ms for 10k documents
- Memory usage: ~1GB per million embeddings

### Cloud API Latency
- Gemini: 1-3 seconds
- Claude: 2-4 seconds
- OpenAI: 1-3 seconds

## Error Handling

### Common Errors

```python
# API key missing
CloudAPIException: "API key not found for provider: gemini"

# Rate limiting
CloudAPIException: "Rate limit exceeded for claude"

# Cost limit reached
DecisionEngineException: "Daily cost limit of $10.00 exceeded"

# Embedding failure
VectorDatabaseException: "Failed to generate embedding for empty text"
```

### Best Practices

1. **Always check API availability** before making cloud requests
2. **Use fallback providers** for critical analyses
3. **Monitor daily costs** to avoid unexpected charges
4. **Cache embeddings** for frequently accessed content
5. **Batch operations** when processing multiple items

## Integration with Other Phases

### Phase 1-3 Integration
- Captures feed into vector database automatically
- OCR text used for embedding generation
- Florence-2 descriptions enhance semantic search

### Phase 5+ Preparation
- Vector database ready for advanced analytics
- Cloud APIs support future agentic capabilities
- NLP foundation for conversational interfaces

## Testing

```bash
# Run Phase 4 validation tests
python test_phase4.py

# Test individual components
python -c "from eidolon.storage.vector_db import VectorDatabase; print('Vector DB OK')"
python -c "from eidolon.models.cloud_api import CloudAPIManager; print('Cloud API OK')"
python -c "from eidolon.core.memory import MemorySystem; print('Memory System OK')"
```

## Security Notes

- API keys stored in environment variables only
- No keys logged or stored in database
- Automatic redaction of sensitive content
- Local-only mode available for privacy