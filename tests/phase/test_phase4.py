#!/usr/bin/env python3
"""
Phase 4 Validation Script for Eidolon AI Personal Assistant

Tests cloud AI integration, vector database, semantic search, natural language queries,
and RAG (Retrieval-Augmented Generation) capabilities.
"""

import os
import sys
import time
import json
import warnings
import asyncio
from pathlib import Path

# Fix tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from datetime import datetime, timedelta

# Suppress known warnings from external libraries
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", message=".*invalid escape sequence.*")

# Set tokenizer parallelism to avoid multiprocessing issues
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.insert(0, '.')


def test_vector_database_initialization():
    """Test vector database and embedding system initialization."""
    print("1️⃣  Testing Vector Database Initialization...")
    print("-" * 50)
    
    try:
        from eidolon.storage.vector_db import VectorDatabase, EmbeddingGenerator
        
        # Test embedding generator
        embedding_gen = EmbeddingGenerator()
        print(f"✅ Embedding generator initialized")
        print(f"   - Model: {embedding_gen.model_name}")
        
        # Test text embedding
        test_text = "This is a test document about Python programming and machine learning"
        embedding = embedding_gen.generate_text_embedding(test_text)
        
        if embedding and len(embedding) > 0:
            print(f"✅ Text embedding generated")
            print(f"   - Embedding dimension: {len(embedding)}")
            print(f"   - Sample values: {embedding[:3]}")
        else:
            print("❌ Failed to generate text embedding")
            return False
        
        # Test vector database
        vector_db = VectorDatabase()
        print(f"✅ Vector database initialized")
        print(f"   - Database path: {vector_db.db_path}")
        print(f"   - Collection name: {vector_db.collection_name}")
        
        # Test content embedding
        test_content = {
            "content_type": "document", 
            "description": "Programming tutorial",
            "tags": ["python", "programming", "tutorial"]
        }
        content_embedding = embedding_gen.generate_content_embedding(test_content)
        
        if content_embedding and len(content_embedding) > 0:
            print(f"✅ Content embedding generated")
            print(f"   - Content embedding dimension: {len(content_embedding)}")
        else:
            print("❌ Failed to generate content embedding")
            return False
        
        print("✅ Vector database initialization successful!")
        return True
        
    except Exception as e:
        print(f"❌ Vector database initialization failed: {e}")
        return False


def test_cloud_api_integrations():
    """Test cloud AI API integrations (without requiring API keys)."""
    print("2️⃣  Testing Cloud AI API Integrations...")
    print("-" * 50)
    
    try:
        from eidolon.models.cloud_api import CloudAPIManager, GeminiAPI, ClaudeAPI, OpenRouterClaudeAPI, OpenAIAPI
        
        # Test individual API classes
        gemini = GeminiAPI()
        claude = ClaudeAPI() 
        openrouter_claude = OpenRouterClaudeAPI()
        openai = OpenAIAPI()
        
        print(f"✅ API classes initialized")
        print(f"   - Gemini available: {gemini.available}")
        print(f"   - Claude available: {claude.available}")
        print(f"   - OpenRouter Claude available: {openrouter_claude.available}")
        print(f"   - OpenAI available: {openai.available}")
        
        # Test API manager
        api_manager = CloudAPIManager()
        print(f"✅ Cloud API manager initialized")
        
        available_providers = api_manager.get_available_providers()
        print(f"   - Available providers: {available_providers}")
        
        # Test usage tracking
        usage_stats = api_manager.get_usage_stats()
        print(f"✅ Usage tracking working")
        print(f"   - Total requests: {usage_stats['total_requests']}")
        print(f"   - Total cost: ${usage_stats['total_cost_usd']}")
        
        print("✅ Cloud API integrations working!")
        return True
        
    except Exception as e:
        print(f"❌ Cloud API integration test failed: {e}")
        return False


def test_decision_engine():
    """Test the local/cloud decision engine."""
    print("3️⃣  Testing Decision Engine...")
    print("-" * 50)
    
    try:
        from eidolon.models.decision_engine import DecisionEngine, AnalysisRequest
        
        # Initialize decision engine
        decision_engine = DecisionEngine()
        print(f"✅ Decision engine initialized")
        print(f"   - Importance threshold: {decision_engine.importance_threshold}")
        print(f"   - Daily cost limit: ${decision_engine.cost_limit_daily}")
        print(f"   - Local first: {decision_engine.local_first}")
        
        # Test decision for simple content
        simple_request = AnalysisRequest(
            content_type="text",
            text_content="Hello world",
            image_size=(800, 600),
            metadata={"content_type": "simple"}
        )
        
        simple_decision = decision_engine.make_routing_decision(simple_request, ["anthropic", "google"])
        print(f"✅ Simple content decision")
        print(f"   - Use cloud: {simple_decision.use_cloud}")
        print(f"   - Reasoning: {simple_decision.reasoning}")
        print(f"   - Confidence: {simple_decision.confidence:.2f}")
        
        # Test decision for complex content
        complex_request = AnalysisRequest(
            content_type="mixed",
            text_content="Complex error analysis with detailed traceback and multiple function calls involving advanced debugging techniques",
            image_size=(2560, 1440),
            metadata={"content_type": "error", "has_ui_elements": True},
            user_preferences={"importance": 0.9}
        )
        
        complex_decision = decision_engine.make_routing_decision(complex_request, ["anthropic", "google"])
        print(f"✅ Complex content decision")
        print(f"   - Use cloud: {complex_decision.use_cloud}")
        print(f"   - Provider: {complex_decision.provider}")
        print(f"   - Reasoning: {complex_decision.reasoning}")
        print(f"   - Estimated quality: {complex_decision.estimated_quality:.2f}")
        
        # Test statistics
        stats = decision_engine.get_decision_stats()
        print(f"✅ Decision statistics")
        print(f"   - Total requests today: {stats['total_requests_today']}")
        print(f"   - Cloud usage: {stats['cloud_usage_percent']:.1f}%")
        
        print("✅ Decision engine working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Decision engine test failed: {e}")
        return False


async def test_openrouter_claude_integration():
    """Test OpenRouter Claude API integration specifically."""
    print("2b️⃣  Testing OpenRouter Claude Integration...")
    print("-" * 50)
    
    try:
        from eidolon.models.cloud_api import OpenRouterClaudeAPI
        
        # Initialize with test key (should fail gracefully without real key)
        openrouter_claude = OpenRouterClaudeAPI()
        print(f"✅ OpenRouter Claude API initialized")
        print(f"   - Available: {openrouter_claude.available}")
        
        # Test the direct call_claude_sonnet method
        if openrouter_claude.available:
            print("✅ Testing direct Claude Sonnet call...")
            test_prompt = "What is the capital of France?"
            response = openrouter_claude.call_claude_sonnet(test_prompt)
            print(f"   - Prompt: {test_prompt}")
            print(f"   - Response preview: {response[:100]}...")
        else:
            print("⚠️  OpenRouter Claude not available (expected without API key)")
            # Test that the class structure is correct
            assert hasattr(openrouter_claude, 'call_claude_sonnet'), "call_claude_sonnet method missing"
            assert hasattr(openrouter_claude, 'analyze_text'), "analyze_text method missing"
            assert hasattr(openrouter_claude, 'analyze_image'), "analyze_image method missing"
            print("✅ All required methods present in OpenRouter Claude API")
        
        # Test that it's properly integrated in the CloudAPIManager
        from eidolon.models.cloud_api import CloudAPIManager
        api_manager = CloudAPIManager()
        
        providers = api_manager.get_available_providers()
        print(f"✅ Available providers: {providers}")
        
        # Check that openrouter_claude is in the usage tracking
        usage_stats = api_manager.get_usage_stats()
        assert 'openrouter_claude' in usage_stats['by_provider'], "OpenRouter Claude not in usage tracking"
        print("✅ OpenRouter Claude properly integrated in CloudAPIManager")
        
        print("✅ OpenRouter Claude integration working!")
        return True
        
    except Exception as e:
        print(f"❌ OpenRouter Claude integration test failed: {e}")
        return False


async def test_semantic_search():
    """Test semantic search capabilities."""
    print("4️⃣  Testing Semantic Search...")
    print("-" * 50)
    
    try:
        from eidolon.storage.vector_db import VectorDatabase
        
        vector_db = VectorDatabase()
        
        # Store some test content
        test_content = [
            {
                "id": "test1",
                "analysis": {
                    "content_type": "development",
                    "description": "Python programming tutorial with Flask web framework",
                    "tags": ["python", "flask", "web", "development"],
                    "confidence": 0.9
                },
                "text": "This tutorial covers Python Flask web development including routing, templates, and database integration."
            },
            {
                "id": "test2", 
                "analysis": {
                    "content_type": "document",
                    "description": "Machine learning research paper about neural networks",
                    "tags": ["machine learning", "neural networks", "research"],
                    "confidence": 0.8
                },
                "text": "Deep learning approaches using convolutional neural networks for image classification tasks."
            },
            {
                "id": "test3",
                "analysis": {
                    "content_type": "terminal",
                    "description": "Command line operations with git version control",
                    "tags": ["git", "terminal", "version control"],
                    "confidence": 0.85
                },
                "text": "git commit -m 'Add new feature' && git push origin main"
            }
        ]
        
        # Store test content
        stored_count = 0
        for content in test_content:
            success = vector_db.store_content(
                content["id"],
                content["analysis"],
                content["text"],
                {"test_data": True}
            )
            if success:
                stored_count += 1
        
        print(f"✅ Stored {stored_count}/{len(test_content)} test documents")
        
        # Test semantic search
        search_queries = [
            "Python web development",
            "machine learning algorithms", 
            "git version control",
            "neural network classification"
        ]
        
        for query in search_queries:
            results = vector_db.semantic_search(query, n_results=3)
            print(f"✅ Search: '{query}'")
            print(f"   - Results: {len(results)}")
            
            for i, result in enumerate(results[:2], 1):
                print(f"   {i}. Similarity: {result['similarity']:.3f} | {result['document'][:50]}...")
        
        # Test hybrid search
        hybrid_results = vector_db.hybrid_search("Python programming", n_results=3)
        print(f"✅ Hybrid search results: {len(hybrid_results)}")
        
        # Test database statistics
        stats = vector_db.get_statistics()
        print(f"✅ Vector database statistics")
        print(f"   - Total documents: {stats['total_documents']}")
        print(f"   - Embedding model: {stats['embedding_model']}")
        
        print("✅ Semantic search working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Semantic search test failed: {e}")
        return False


async def test_natural_language_processing():
    """Test natural language query processing."""
    print("5️⃣  Testing Natural Language Processing...")
    print("-" * 50)
    
    try:
        from eidolon.core.memory import MemorySystem
        
        memory_system = MemorySystem()
        print(f"✅ Memory system initialized")
        
        # Test query intent parsing
        test_queries = [
            "Find all Python code from yesterday",
            "Summarize my work on machine learning",
            "Show me terminal commands from this week",
            "What errors did I encounter today?",
            "Compare my productivity this month vs last month"
        ]
        
        for query in test_queries:
            intent = memory_system.parse_natural_language_query(query)
            print(f"✅ Query: '{query}'")
            print(f"   - Intent: {intent.intent_type}")
            print(f"   - Search terms: {intent.search_terms}")
            print(f"   - Filters: {intent.filters}")
            print(f"   - Confidence: {intent.confidence:.2f}")
            
            if intent.time_range:
                print(f"   - Time range: {intent.time_range}")
            print()
        
        print("✅ Natural language processing working!")
        return True
        
    except Exception as e:
        print(f"❌ Natural language processing test failed: {e}")
        return False


async def test_memory_system_integration():
    """Test complete memory system with semantic search and storage."""
    print("6️⃣  Testing Memory System Integration...")
    print("-" * 50)
    
    try:
        from eidolon.core.memory import MemorySystem
        
        memory_system = MemorySystem()
        
        # Test content storage
        test_screenshots = [
            {
                "id": "mem_test1",
                "analysis": {
                    "content_type": "development",
                    "description": "VS Code with Python debugging session",
                    "confidence": 0.9,
                    "tags": ["python", "debugging", "vscode"],
                    "vision_analysis": {
                        "scene_type": "development",
                        "description": "Code editor showing Python debugging interface",
                        "model_used": "florence-2"
                    }
                },
                "text": "def calculate_fibonacci(n): if n <= 1: return n else: return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)"
            },
            {
                "id": "mem_test2",
                "analysis": {
                    "content_type": "browser",
                    "description": "Stack Overflow page about Python optimization",
                    "confidence": 0.8,
                    "tags": ["python", "optimization", "stackoverflow"]
                },
                "text": "How to optimize Python code performance using profiling and caching techniques"
            }
        ]
        
        stored_count = 0
        for screenshot in test_screenshots:
            success = memory_system.store_content(
                screenshot["id"],
                screenshot["analysis"],
                screenshot["text"],
                {"integration_test": True}
            )
            if success:
                stored_count += 1
        
        print(f"✅ Stored {stored_count}/{len(test_screenshots)} items in memory system")
        
        # Test semantic search through memory system
        search_results = await memory_system.semantic_search(
            "Python code optimization",
            n_results=5
        )
        
        print(f"✅ Memory semantic search")
        print(f"   - Results found: {len(search_results)}")
        
        for result in search_results[:2]:
            print(f"   - {result.content_id}: {result.similarity_score:.3f}")
            print(f"     {result.content[:80]}...")
        
        # Test memory statistics
        stats = memory_system.get_memory_statistics()
        print(f"✅ Memory system statistics")
        print(f"   - Searchable content: {stats.get('total_searchable_content', 0)}")
        print(f"   - Capabilities: {list(stats.get('capabilities', {}).keys())}")
        
        print("✅ Memory system integration working!")
        return True
        
    except Exception as e:
        print(f"❌ Memory system integration test failed: {e}")
        return False


async def test_rag_system():
    """Test RAG (Retrieval-Augmented Generation) system."""
    print("7️⃣  Testing RAG System...")
    print("-" * 50)
    
    try:
        from eidolon.core.memory import MemorySystem
        
        memory_system = MemorySystem()
        
        # Test end-to-end natural language query processing
        test_queries = [
            "What Python code have I been working on?",
            "Summarize my recent programming activities",
            "Find debugging sessions from today"
        ]
        
        for query in test_queries:
            print(f"🤖 Processing query: '{query}'")
            
            response = await memory_system.process_natural_language_query(query)
            
            print(f"✅ Query processed")
            print(f"   - Intent: {response.query_intent.intent_type}")
            print(f"   - Search results: {len(response.search_results)}")
            print(f"   - Generated by: {response.generated_by}")
            print(f"   - Confidence: {response.confidence:.2f}")
            print(f"   - Response preview: {response.response[:100]}...")
            print()
        
        # Test basic RAG response generation
        from eidolon.core.memory import SearchResult
        
        mock_results = [
            SearchResult(
                content_id="rag_test1",
                content="Python function for calculating Fibonacci numbers with recursive approach",
                similarity_score=0.85,
                timestamp=datetime.now(),
                metadata={"content_type": "development"}
            ),
            SearchResult(
                content_id="rag_test2", 
                content="Code optimization techniques using profiling and memoization",
                similarity_score=0.78,
                timestamp=datetime.now(),
                metadata={"content_type": "development"}
            )
        ]
        
        rag_response = await memory_system.generate_rag_response(
            "What optimization techniques did I use?",
            mock_results
        )
        
        print(f"✅ RAG response generated")
        print(f"   - Response length: {len(rag_response)} characters")
        print(f"   - Response preview: {rag_response[:150]}...")
        
        print("✅ RAG system working!")
        return True
        
    except Exception as e:
        print(f"❌ RAG system test failed: {e}")
        return False


async def main():
    """Run all Phase 4 validation tests."""
    print("\n🤖 EIDOLON PHASE 4 VALIDATION SCRIPT")
    print("=" * 50)
    print("Testing Cloud AI & Semantic Memory capabilities\n")
    
    tests = [
        ("Vector Database Initialization", test_vector_database_initialization),
        ("Cloud AI Integrations", test_cloud_api_integrations),  
        ("OpenRouter Claude Integration", test_openrouter_claude_integration),
        ("Decision Engine", test_decision_engine),
        ("Semantic Search", test_semantic_search),
        ("Natural Language Processing", test_natural_language_processing),
        ("Memory System Integration", test_memory_system_integration),
        ("RAG System", test_rag_system)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
                
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            failed += 1
        print()
    
    print("=" * 50)
    print("📊 PHASE 4 VALIDATION SUMMARY")
    print("=" * 50)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 ALL PHASE 4 TESTS PASSED!")
        print("\n📝 Phase 4 features working:")
        print("• Vector database with ChromaDB")
        print("• Semantic search and embeddings")
        print("• Cloud AI API integrations (Gemini, Claude, OpenRouter-Claude, OpenAI)")
        print("• OpenRouter.ai integration for cost-effective Claude access")
        print("• Intelligent local/cloud routing")
        print("• Natural language query processing")
        print("• RAG (Retrieval-Augmented Generation)")
        print("• Hybrid search capabilities")
        print("• Intent recognition and parsing")
        
        print("\n🚀 Ready for Phase 5: Advanced Analytics!")
    else:
        print(f"\n⚠️  {failed} tests failed. Check output above.")
        print("\nNote: Some tests may fail without API keys, but core functionality should work.")
        print("Cloud AI features require valid API keys in environment variables.")
    
    print()


if __name__ == "__main__":
    asyncio.run(main())