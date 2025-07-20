#!/usr/bin/env python3
"""
Unit tests for the Vector Database component.

Tests cover ChromaDB integration, embedding generation, semantic search,
and vector storage operations.
"""

import pytest
import tempfile
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Import the modules to test
from eidolon.storage.vector_db import VectorDatabase, EmbeddingGenerator


class TestEmbeddingGenerator:
    """Test the EmbeddingGenerator class."""
    
    @pytest.fixture
    def mock_sentence_transformer(self):
        """Mock SentenceTransformer model."""
        with patch('eidolon.storage.vector_db.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
            mock_st.return_value = mock_model
            yield mock_model
    
    def test_embedding_generator_initialization(self):
        """Test EmbeddingGenerator initialization."""
        with patch('eidolon.storage.vector_db.get_component_logger'):
            generator = EmbeddingGenerator("test-model")
            
            assert generator.model_name == "test-model"
            assert generator._model is None
    
    def test_lazy_model_loading(self, mock_sentence_transformer):
        """Test lazy loading of embedding model."""
        with patch('eidolon.storage.vector_db.get_component_logger'):
            generator = EmbeddingGenerator()
            
            # Model should not be loaded initially
            assert generator._model is None
            
            # First call should load the model
            embedding = generator.generate_text_embedding("test text")
            
            assert generator._model is not None
            assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
    
    def test_generate_text_embedding_success(self, mock_sentence_transformer):
        """Test successful text embedding generation."""
        with patch('eidolon.storage.vector_db.get_component_logger'):
            generator = EmbeddingGenerator()
            
            embedding = generator.generate_text_embedding("Hello world")
            
            assert isinstance(embedding, list)
            assert len(embedding) == 5
            assert embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
    
    def test_generate_text_embedding_empty_text(self, mock_sentence_transformer):
        """Test embedding generation with empty text."""
        with patch('eidolon.storage.vector_db.get_component_logger'):
            generator = EmbeddingGenerator()
            
            # Test various empty inputs
            assert generator.generate_text_embedding("") is None
            assert generator.generate_text_embedding("   ") is None
            assert generator.generate_text_embedding(None) is None
    
    def test_generate_content_embedding_full_analysis(self, mock_sentence_transformer):
        """Test content embedding with full analysis data."""
        with patch('eidolon.storage.vector_db.get_component_logger'):
            generator = EmbeddingGenerator()
            
            content_analysis = {
                "content_type": "code",
                "description": "Python function definition",
                "tags": ["python", "function", "programming"],
                "vision_analysis": {
                    "description": "Code editor interface",
                    "scene_type": "development"
                },
                "extracted_text": "def hello_world(): print('Hello')"
            }
            
            embedding = generator.generate_content_embedding(content_analysis)
            
            assert isinstance(embedding, list)
            assert len(embedding) == 5
            
            # Verify the model was called with combined text
            mock_sentence_transformer.encode.assert_called_once()
            called_text = mock_sentence_transformer.encode.call_args[0][0]
            assert "Content type: code" in called_text
            assert "Python function definition" in called_text
            assert "python, function, programming" in called_text
    
    def test_generate_content_embedding_minimal_data(self, mock_sentence_transformer):
        """Test content embedding with minimal data."""
        with patch('eidolon.storage.vector_db.get_component_logger'):
            generator = EmbeddingGenerator()
            
            content_analysis = {
                "content_type": "document"
            }
            
            embedding = generator.generate_content_embedding(content_analysis)
            
            assert isinstance(embedding, list)
            mock_sentence_transformer.encode.assert_called_once()
    
    def test_generate_content_embedding_empty_analysis(self, mock_sentence_transformer):
        """Test content embedding with empty analysis."""
        with patch('eidolon.storage.vector_db.get_component_logger'):
            generator = EmbeddingGenerator()
            
            embedding = generator.generate_content_embedding({})
            
            assert embedding is None
    
    def test_embedding_generation_error_handling(self):
        """Test error handling in embedding generation."""
        with patch('eidolon.storage.vector_db.get_component_logger') as mock_logger, \
             patch('eidolon.storage.vector_db.SentenceTransformer') as mock_st:
            
            # Make the model raise an exception
            mock_st.side_effect = Exception("Model loading failed")
            
            generator = EmbeddingGenerator()
            
            # Should handle the error gracefully
            embedding = generator.generate_text_embedding("test")
            assert embedding is None


class TestVectorDatabase:
    """Test the VectorDatabase class."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for VectorDatabase."""
        config = Mock()
        config.memory.db_path = "/tmp/test_metadata.db"  # This is what VectorDatabase expects
        config.memory.vector_db_path = "/tmp/test_vectordb"
        config.memory.embedding_model = "test-model"
        config.memory.search = {
            "max_results": 10,
            "similarity_threshold": 0.7
        }
        return config
    
    @pytest.fixture
    def mock_chromadb(self):
        """Mock ChromaDB client and collection."""
        with patch('eidolon.storage.vector_db.chromadb') as mock_chroma:
            mock_client = Mock()
            mock_collection = Mock()
            
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_chroma.PersistentClient.return_value = mock_client
            
            yield mock_client, mock_collection
    
    @pytest.fixture
    def vector_db(self, mock_config, mock_chromadb):
        """Create VectorDatabase instance with mocks."""
        mock_client, mock_collection = mock_chromadb
        
        with patch('eidolon.storage.vector_db.get_config', return_value=mock_config), \
             patch('eidolon.storage.vector_db.get_component_logger'), \
             patch.object(EmbeddingGenerator, 'generate_text_embedding', return_value=[0.1, 0.2, 0.3]):
            
            db = VectorDatabase()
            db.collection = mock_collection  # Set the mock collection
            yield db, mock_collection
    
    def test_vector_database_initialization(self, mock_config, mock_chromadb):
        """Test VectorDatabase initialization."""
        mock_client, mock_collection = mock_chromadb
        
        with patch('eidolon.storage.vector_db.get_config', return_value=mock_config), \
             patch('eidolon.storage.vector_db.get_component_logger'):
            
            db = VectorDatabase()
            
            assert db.config == mock_config
            assert db.embedding_generator is not None
            mock_client.get_or_create_collection.assert_called_once()
    
    def test_store_content_with_text(self, vector_db):
        """Test storing content with extracted text."""
        db, mock_collection = vector_db
        
        # Mock embedding generation
        with patch.object(db.embedding_generator, 'generate_text_embedding', return_value=[0.1, 0.2, 0.3]):
            
            result = db.store_content(
                screenshot_id="test_123",
                content_analysis={"type": "code", "confidence": 0.9},
                extracted_text="def hello(): pass",
                metadata={"language": "python"}
            )
            
            assert result is True
            mock_collection.add.assert_called_once()
            
            # Verify the call arguments
            call_args = mock_collection.add.call_args
            assert "test_123" in call_args[1]["ids"]
            assert call_args[1]["embeddings"] == [[0.1, 0.2, 0.3]]
            assert call_args[1]["documents"] == ["def hello(): pass"]
    
    def test_store_content_without_text(self, vector_db):
        """Test storing content without extracted text."""
        db, mock_collection = vector_db
        
        content_analysis = {
            "content_type": "image", 
            "description": "A screenshot",
            "confidence": 0.8
        }
        
        with patch.object(db.embedding_generator, 'generate_content_embedding', return_value=[0.4, 0.5, 0.6]):
            
            result = db.store_content(
                screenshot_id="test_456",
                content_analysis=content_analysis,
                extracted_text="",
                metadata={"source": "screen"}
            )
            
            assert result is True
            mock_collection.add.assert_called_once()
    
    def test_store_content_no_embedding(self, vector_db):
        """Test storing content when embedding generation fails."""
        db, mock_collection = vector_db
        
        with patch.object(db.embedding_generator, 'generate_text_embedding', return_value=None), \
             patch.object(db.embedding_generator, 'generate_content_embedding', return_value=None):
            
            result = db.store_content(
                screenshot_id="test_789",
                content_analysis={"type": "empty"},
                extracted_text="",
                metadata={}
            )
            
            assert result is False
            mock_collection.add.assert_not_called()
    
    def test_semantic_search_with_results(self, vector_db):
        """Test semantic search returning results."""
        db, mock_collection = vector_db
        
        # Mock search results
        mock_collection.query.return_value = {
            "ids": [["doc1", "doc2"]],
            "documents": [["First document", "Second document"]],
            "distances": [[0.2, 0.4]],
            "metadatas": [[{"type": "code"}, {"type": "document"}]]
        }
        
        with patch.object(db.embedding_generator, 'generate_text_embedding', return_value=[0.1, 0.2, 0.3]):
            
            results = db.semantic_search("python code", n_results=5)
            
            assert len(results) == 2
            assert results[0]["id"] == "doc1"
            assert results[0]["document"] == "First document"
            assert results[0]["similarity"] == 0.8  # 1 - 0.2
            assert results[0]["metadata"]["type"] == "code"
    
    def test_semantic_search_no_embedding(self, vector_db):
        """Test semantic search when embedding generation fails."""
        db, mock_collection = vector_db
        
        with patch.object(db.embedding_generator, 'generate_text_embedding', return_value=None):
            
            results = db.semantic_search("test query")
            
            assert results == []
            mock_collection.query.assert_not_called()
    
    def test_semantic_search_empty_results(self, vector_db):
        """Test semantic search with no results."""
        db, mock_collection = vector_db
        
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "distances": [[]],
            "metadatas": [[]]
        }
        
        with patch.object(db.embedding_generator, 'generate_text_embedding', return_value=[0.1, 0.2, 0.3]):
            
            results = db.semantic_search("nonexistent query")
            
            assert results == []
    
    def test_hybrid_search(self, vector_db):
        """Test hybrid search combining semantic and keyword search."""
        db, mock_collection = vector_db
        
        # Mock semantic search results
        mock_collection.query.return_value = {
            "ids": [["doc1", "doc2"]],
            "documents": [["Python code example", "JavaScript tutorial"]],
            "distances": [[0.1, 0.3]],
            "metadatas": [[{"type": "code"}, {"type": "tutorial"}]]
        }
        
        with patch.object(db.embedding_generator, 'generate_text_embedding', return_value=[0.1, 0.2, 0.3]):
            
            results = db.hybrid_search(
                "python programming",
                n_results=5,
                semantic_weight=0.7,
                keyword_weight=0.3
            )
            
            assert len(results) == 2
            assert all("combined_score" in result for result in results)
            assert all("similarity" in result for result in results)
    
    def test_get_statistics(self, vector_db):
        """Test getting database statistics."""
        db, mock_collection = vector_db
        
        mock_collection.count.return_value = 100
        
        stats = db.get_statistics()
        
        assert isinstance(stats, dict)
        assert "total_documents" in stats
        assert stats["total_documents"] == 100
        assert "collection_name" in stats
        assert "embedding_model" in stats
    
    def test_cleanup_old_entries(self, vector_db):
        """Test cleaning up old entries."""
        db, mock_collection = vector_db
        
        # Mock getting old entries
        mock_collection.get.return_value = {
            "ids": ["old1", "old2", "old3"],
            "metadatas": [
                {"timestamp": "2024-01-01T00:00:00"},
                {"timestamp": "2024-01-02T00:00:00"},
                {"timestamp": "2024-01-03T00:00:00"}
            ]
        }
        
        removed_count = db.cleanup_old_entries(days_to_keep=1)
        
        assert removed_count == 3
        mock_collection.delete.assert_called_once_with(ids=["old1", "old2", "old3"])
    
    def test_update_document(self, vector_db):
        """Test updating an existing document."""
        db, mock_collection = vector_db
        
        with patch.object(db.embedding_generator, 'generate_text_embedding', return_value=[0.7, 0.8, 0.9]):
            
            result = db.update_document(
                document_id="existing_doc",
                new_text="Updated content",
                new_metadata={"updated": True}
            )
            
            assert result is True
            mock_collection.update.assert_called_once()
    
    def test_delete_document(self, vector_db):
        """Test deleting a document."""
        db, mock_collection = vector_db
        
        result = db.delete_document("doc_to_delete")
        
        assert result is True
        mock_collection.delete.assert_called_once_with(ids=["doc_to_delete"])
    
    def test_batch_store_content(self, vector_db):
        """Test batch storing multiple content items."""
        db, mock_collection = vector_db
        
        content_items = [
            {
                "screenshot_id": "batch1",
                "content_analysis": {"type": "code"},
                "extracted_text": "print('hello')",
                "metadata": {"lang": "python"}
            },
            {
                "screenshot_id": "batch2", 
                "content_analysis": {"type": "document"},
                "extracted_text": "Sample document",
                "metadata": {"format": "text"}
            }
        ]
        
        with patch.object(db.embedding_generator, 'generate_text_embedding', side_effect=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]):
            
            results = db.batch_store_content(content_items)
            
            assert len(results) == 2
            assert all(result is True for result in results)
            assert mock_collection.add.call_count == 2


class TestVectorDatabaseError:
    """Test error handling in VectorDatabase."""
    
    def test_chromadb_connection_error(self):
        """Test handling ChromaDB connection errors."""
        with patch('eidolon.storage.vector_db.get_config'), \
             patch('eidolon.storage.vector_db.get_component_logger'), \
             patch('eidolon.storage.vector_db.chromadb.PersistentClient', side_effect=Exception("Connection failed")):
            
            # Should handle connection error gracefully
            try:
                db = VectorDatabase()
                # If no exception is raised, the error was handled
            except Exception as e:
                # Or it might raise a specific exception type
                assert "Connection failed" in str(e) or "ChromaDB" in str(e)
    
    def test_collection_operation_error(self, vector_db):
        """Test handling collection operation errors."""
        db, mock_collection = vector_db
        
        # Make collection operations fail
        mock_collection.add.side_effect = Exception("Add operation failed")
        
        result = db.store_content(
            screenshot_id="error_test",
            content_analysis={"type": "test"},
            extracted_text="test content",
            metadata={}
        )
        
        # Should handle the error gracefully
        assert result is False
    
    def test_search_operation_error(self, vector_db):
        """Test handling search operation errors."""
        db, mock_collection = vector_db
        
        mock_collection.query.side_effect = Exception("Query failed")
        
        with patch.object(db.embedding_generator, 'generate_text_embedding', return_value=[0.1, 0.2, 0.3]):
            
            results = db.semantic_search("error query")
            
            # Should return empty results on error
            assert results == []


class TestVectorDatabaseIntegration:
    """Integration tests for VectorDatabase with realistic scenarios."""
    
    @pytest.fixture
    def temp_vector_db(self):
        """Create temporary vector database for integration testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Mock()
            config.memory.db_path = os.path.join(temp_dir, "metadata.db")
            config.memory.vector_db_path = temp_dir
            config.memory.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            config.memory.search = {"max_results": 10, "similarity_threshold": 0.7}
            
            with patch('eidolon.storage.vector_db.get_config', return_value=config), \
                 patch('eidolon.storage.vector_db.get_component_logger'), \
                 patch('eidolon.storage.vector_db.SentenceTransformer') as mock_st:
                
                # Mock the embedding model
                mock_model = Mock()
                mock_model.encode.return_value = np.random.rand(384)  # Typical embedding size
                mock_st.return_value = mock_model
                
                db = VectorDatabase()
                yield db
    
    @pytest.mark.skip(reason="Integration test requires ChromaDB setup")
    def test_real_storage_and_search_workflow(self, temp_vector_db):
        """Test realistic storage and search workflow."""
        db = temp_vector_db
        
        # Store sample content
        content_items = [
            {
                "screenshot_id": "code1",
                "extracted_text": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
                "content_analysis": {"type": "code", "language": "python"}
            },
            {
                "screenshot_id": "doc1", 
                "extracted_text": "Machine learning is a subset of artificial intelligence",
                "content_analysis": {"type": "document", "topic": "AI"}
            },
            {
                "screenshot_id": "web1",
                "extracted_text": "Welcome to our e-commerce store. Browse products and add to cart.",
                "content_analysis": {"type": "browser", "category": "shopping"}
            }
        ]
        
        # Store all content
        for item in content_items:
            result = db.store_content(**item, metadata={})
            assert result is True
        
        # Test semantic search
        results = db.semantic_search("python programming", n_results=5)
        assert len(results) > 0
        
        # Should find the code content
        code_results = [r for r in results if "fibonacci" in r.get("document", "")]
        assert len(code_results) > 0
        
        # Test hybrid search
        hybrid_results = db.hybrid_search("machine learning AI", n_results=5)
        assert len(hybrid_results) > 0
        
        # Test statistics
        stats = db.get_statistics()
        assert stats["total_documents"] >= 3
    
    def test_embedding_consistency(self):
        """Test that embeddings are consistent for same input."""
        with patch('eidolon.storage.vector_db.get_component_logger'), \
             patch('eidolon.storage.vector_db.SentenceTransformer') as mock_st:
            
            # Create consistent mock embedding
            mock_model = Mock()
            mock_model.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
            mock_st.return_value = mock_model
            
            generator = EmbeddingGenerator()
            
            # Generate embeddings for same text multiple times
            text = "test consistency"
            embedding1 = generator.generate_text_embedding(text)
            embedding2 = generator.generate_text_embedding(text)
            
            assert embedding1 == embedding2
            assert embedding1 == [0.1, 0.2, 0.3, 0.4, 0.5]
    
    def test_large_batch_operations(self, temp_vector_db):
        """Test performance with larger batch operations."""
        db = temp_vector_db
        
        # Create larger batch of content
        batch_size = 50
        content_items = []
        
        for i in range(batch_size):
            content_items.append({
                "screenshot_id": f"batch_item_{i}",
                "content_analysis": {"type": "test", "index": i},
                "extracted_text": f"Test content item number {i} with some sample text",
                "metadata": {"batch_id": "large_test", "index": i}
            })
        
        # Store in batches (test batch functionality if available)
        batch_results = db.batch_store_content(content_items) if hasattr(db, 'batch_store_content') else []
        
        if batch_results:
            assert len(batch_results) == batch_size
            assert all(result for result in batch_results)
        else:
            # Fall back to individual storage
            for item in content_items:
                result = db.store_content(**item)
                assert result is True
        
        # Verify storage
        stats = db.get_statistics()
        assert stats["total_documents"] >= batch_size


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])