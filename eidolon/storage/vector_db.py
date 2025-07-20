"""
Vector Database integration for semantic search using ChromaDB

Provides semantic search capabilities for screenshots, extracted text,
and AI analysis results using vector embeddings.
"""

import os
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import uuid

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from ..utils.logging import get_component_logger, log_performance, log_exceptions
from ..utils.config import get_config


class EmbeddingGenerator:
    """Generates embeddings for text and image content."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.logger = get_component_logger("embedding_generator")
        self.model_name = model_name
        self._model = None
        
    def _load_model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                self.logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                self.logger.info("Embedding model loaded successfully")
            except (OSError, ValueError, ImportError) as e:
                self.logger.error(f"Failed to load embedding model: {e}")
                raise
            except Exception as e:
                self.logger.error(f"Unexpected error loading embedding model: {e}")
                raise
        return self._model
    
    @log_performance
    def generate_text_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text content."""
        if not text or not text.strip():
            return None
            
        try:
            model = self._load_model()
            embedding = model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except (RuntimeError, ValueError, AttributeError) as e:
            self.logger.error(f"Failed to generate text embedding: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error generating text embedding: {e}")
            return None
    
    @log_performance 
    def generate_content_embedding(self, content_analysis: Dict[str, Any]) -> Optional[List[float]]:
        """Generate embedding for content analysis results."""
        try:
            # Combine various content fields for embedding
            text_parts = []
            
            # Add content type and description
            if content_analysis.get("content_type"):
                text_parts.append(f"Content type: {content_analysis['content_type']}")
            
            if content_analysis.get("description"):
                text_parts.append(content_analysis["description"])
            
            # Add tags
            if content_analysis.get("tags"):
                text_parts.append(f"Tags: {', '.join(content_analysis['tags'])}")
            
            # Add vision analysis if available
            if content_analysis.get("vision_analysis"):
                va = content_analysis["vision_analysis"]
                if va.get("description"):
                    text_parts.append(f"Vision: {va['description']}")
                if va.get("scene_type"):
                    text_parts.append(f"Scene: {va['scene_type']}")
            
            # Add extracted text if available
            if content_analysis.get("extracted_text"):
                text_parts.append(content_analysis["extracted_text"][:500])  # Limit text length
            
            if not text_parts:
                return None
            
            combined_text = " | ".join(text_parts)
            return self.generate_text_embedding(combined_text)
            
        except Exception as e:
            self.logger.error(f"Failed to generate content embedding: {e}")
            return None


class VectorDatabase:
    """ChromaDB-based vector database for semantic search."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.config = get_config()
        self.logger = get_component_logger("storage.vector_db")
        
        # Database configuration
        if db_path is None:
            db_path = os.path.join(os.path.dirname(self.config.memory.db_path), "vector_db")
        
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB
        self._client = None
        self._collection = None
        self.collection_name = "eidolon_screenshots"
        
        # Initialize embedding generator
        embedding_model = self.config.memory.embedding_model
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        
        self.logger.info(f"Vector database initialized at: {self.db_path}")
    
    def _get_client(self):
        """Get or create ChromaDB client."""
        if self._client is None:
            try:
                self._client = chromadb.PersistentClient(path=str(self.db_path))
                self.logger.info("ChromaDB client connected")
            except Exception as e:
                self.logger.error(f"Failed to connect to ChromaDB: {e}")
                raise
        return self._client
    
    def _get_collection(self):
        """Get or create the main collection."""
        if self._collection is None:
            try:
                client = self._get_client()
                
                # Try to get existing collection
                try:
                    self._collection = client.get_collection(self.collection_name)
                    self.logger.info(f"Connected to existing collection: {self.collection_name}")
                except Exception:
                    # Create new collection if it doesn't exist
                    self._collection = client.create_collection(
                        name=self.collection_name,
                        metadata={"description": "Eidolon screenshot and content embeddings"}
                    )
                    self.logger.info(f"Created new collection: {self.collection_name}")
                    
            except Exception as e:
                self.logger.error(f"Failed to get/create collection: {e}")
                raise
        return self._collection
    
    @log_performance
    @log_exceptions("eidolon.vector_db")
    def store_content(
        self, 
        screenshot_id: str, 
        content_analysis: Dict[str, Any],
        extracted_text: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store content analysis with vector embedding.
        
        Args:
            screenshot_id: Unique identifier for the screenshot
            content_analysis: Content analysis results
            extracted_text: OCR extracted text
            metadata: Additional metadata
            
        Returns:
            bool: True if stored successfully
        """
        try:
            collection = self._get_collection()
            
            # Generate embedding for the content
            embedding_content = content_analysis.copy()
            if extracted_text:
                embedding_content["extracted_text"] = extracted_text
            
            embedding = self.embedding_generator.generate_content_embedding(embedding_content)
            if embedding is None:
                self.logger.warning(f"Failed to generate embedding for screenshot {screenshot_id}")
                return False
            
            # Prepare document text for storage
            doc_parts = []
            if extracted_text:
                doc_parts.append(extracted_text)
            if content_analysis.get("description"):
                doc_parts.append(content_analysis["description"])
            
            document = " | ".join(doc_parts) if doc_parts else "No content"
            
            # Prepare metadata
            store_metadata = {
                "screenshot_id": screenshot_id,
                "content_type": content_analysis.get("content_type", "unknown"),
                "confidence": float(content_analysis.get("confidence", 0.0)),
                "timestamp": datetime.now().isoformat(),
                "has_text": bool(extracted_text.strip()),
                "has_vision_analysis": bool(content_analysis.get("vision_analysis")),
                "word_count": len(extracted_text.split()) if extracted_text else 0
            }
            
            # Add tags
            if content_analysis.get("tags"):
                store_metadata["tags"] = ",".join(content_analysis["tags"])
            
            # Add vision analysis metadata
            if content_analysis.get("vision_analysis"):
                va = content_analysis["vision_analysis"]
                store_metadata["scene_type"] = va.get("scene_type", "")
                store_metadata["vision_model"] = va.get("model_used", "")
                store_metadata["vision_confidence"] = float(va.get("confidence", 0.0))
            
            # Add any custom metadata
            if metadata:
                store_metadata.update(metadata)
            
            # Generate unique ID for the vector entry
            vector_id = f"screenshot_{screenshot_id}_{hashlib.md5(document.encode()).hexdigest()[:8]}"
            
            # Store in ChromaDB
            collection.upsert(
                ids=[vector_id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[store_metadata]
            )
            
            self.logger.debug(f"Stored content embedding for screenshot {screenshot_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store content in vector database: {e}")
            return False
    
    @log_performance
    def semantic_search(
        self,
        query: str,
        n_results: int = 10,
        content_type_filter: Optional[str] = None,
        min_confidence: float = 0.0,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search on stored content.
        
        Args:
            query: Search query text
            n_results: Maximum number of results to return
            content_type_filter: Filter by content type
            min_confidence: Minimum confidence threshold
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of search results with similarity scores
        """
        try:
            collection = self._get_collection()
            
            # Generate embedding for the query
            query_embedding = self.embedding_generator.generate_text_embedding(query)
            if query_embedding is None:
                self.logger.warning("Failed to generate embedding for search query")
                return []
            
            # Prepare filters
            where_filter = {}
            if content_type_filter:
                where_filter["content_type"] = content_type_filter
            if min_confidence > 0:
                where_filter["confidence"] = {"$gte": min_confidence}
            
            # Perform search
            search_kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": min(n_results, 100),  # ChromaDB limit
            }
            
            if where_filter:
                search_kwargs["where"] = where_filter
            
            if include_metadata:
                search_kwargs["include"] = ["documents", "metadatas", "distances"]
            else:
                search_kwargs["include"] = ["documents", "distances"]
            
            results = collection.query(**search_kwargs)
            
            # Format results
            formatted_results = []
            if results and results.get("ids") and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    result = {
                        "id": results["ids"][0][i],
                        "document": results["documents"][0][i],
                        "similarity": 1.0 - results["distances"][0][i],  # Convert distance to similarity
                        "distance": results["distances"][0][i]
                    }
                    
                    if include_metadata and results.get("metadatas"):
                        result["metadata"] = results["metadatas"][0][i]
                    
                    formatted_results.append(result)
            
            self.logger.debug(f"Semantic search for '{query}' returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return []
    
    @log_performance
    def hybrid_search(
        self,
        query: str,
        n_results: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        **filters
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword matching.
        
        Args:
            query: Search query
            n_results: Number of results to return
            semantic_weight: Weight for semantic similarity
            keyword_weight: Weight for keyword matching
            **filters: Additional filters
            
        Returns:
            List of search results with combined scores
        """
        try:
            # Get semantic search results
            semantic_results = self.semantic_search(
                query, 
                n_results=n_results * 2,  # Get more to allow for re-ranking
                **filters
            )
            
            # Simple keyword matching score
            query_words = set(query.lower().split())
            
            # Combine and re-rank results
            final_results = []
            for result in semantic_results:
                document_words = set(result["document"].lower().split())
                keyword_overlap = len(query_words.intersection(document_words)) / max(len(query_words), 1)
                
                # Combine scores
                combined_score = (
                    semantic_weight * result["similarity"] + 
                    keyword_weight * keyword_overlap
                )
                
                result["combined_score"] = combined_score
                result["keyword_score"] = keyword_overlap
                final_results.append(result)
            
            # Sort by combined score and return top results
            final_results.sort(key=lambda x: x["combined_score"], reverse=True)
            return final_results[:n_results]
            
        except Exception as e:
            self.logger.error(f"Hybrid search failed: {e}")
            return []
    
    def get_similar_content(
        self,
        screenshot_id: str,
        n_results: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Find content similar to a specific screenshot.
        
        Args:
            screenshot_id: ID of the reference screenshot
            n_results: Number of similar items to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of similar content items
        """
        try:
            collection = self._get_collection()
            
            # Find the reference document
            ref_results = collection.get(
                where={"screenshot_id": screenshot_id},
                include=["embeddings", "documents", "metadatas"]
            )
            
            if not ref_results["ids"]:
                self.logger.warning(f"Screenshot {screenshot_id} not found in vector database")
                return []
            
            # Use the first embedding as reference
            ref_embedding = ref_results["embeddings"][0]
            
            # Search for similar items
            similar_results = collection.query(
                query_embeddings=[ref_embedding],
                n_results=n_results + 1,  # +1 to exclude the reference itself
                include=["documents", "metadatas", "distances"]
            )
            
            # Filter out the reference document and apply similarity threshold
            filtered_results = []
            if similar_results and similar_results.get("ids"):
                for i, result_id in enumerate(similar_results["ids"][0]):
                    # Skip the reference document itself
                    if similar_results["metadatas"][0][i]["screenshot_id"] == screenshot_id:
                        continue
                    
                    similarity = 1.0 - similar_results["distances"][0][i]
                    if similarity >= similarity_threshold:
                        filtered_results.append({
                            "id": result_id,
                            "document": similar_results["documents"][0][i],
                            "similarity": similarity,
                            "metadata": similar_results["metadatas"][0][i]
                        })
            
            return filtered_results[:n_results]
            
        except Exception as e:
            self.logger.error(f"Failed to find similar content: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector database statistics."""
        try:
            collection = self._get_collection()
            count = collection.count()
            
            # Get sample of metadata for analysis
            sample_results = collection.get(
                limit=min(100, count),
                include=["metadatas"]
            )
            
            stats = {
                "total_documents": count,
                "embedding_model": self.embedding_generator.model_name,
                "collection_name": self.collection_name,
                "db_path": str(self.db_path)
            }
            
            # Analyze content types and other metadata
            if sample_results.get("metadatas"):
                content_types = {}
                scene_types = {}
                has_vision = 0
                has_text = 0
                
                for metadata in sample_results["metadatas"]:
                    # Content types
                    ct = metadata.get("content_type", "unknown")
                    content_types[ct] = content_types.get(ct, 0) + 1
                    
                    # Scene types
                    st = metadata.get("scene_type", "")
                    if st:
                        scene_types[st] = scene_types.get(st, 0) + 1
                    
                    # Vision and text analysis
                    if metadata.get("has_vision_analysis"):
                        has_vision += 1
                    if metadata.get("has_text"):
                        has_text += 1
                
                stats.update({
                    "content_types": content_types,
                    "scene_types": scene_types,
                    "documents_with_vision": has_vision,
                    "documents_with_text": has_text,
                    "sample_size": len(sample_results["metadatas"])
                })
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get vector database statistics: {e}")
            return {"error": str(e)}
    
    def cleanup_old_entries(self, days_to_keep: int = 30) -> int:
        """
        Remove old entries from the vector database.
        
        Args:
            days_to_keep: Number of days of data to keep
            
        Returns:
            Number of entries removed
        """
        try:
            from datetime import datetime, timedelta
            
            collection = self._get_collection()
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
            
            # Get entries older than cutoff
            old_entries = collection.get(
                where={"timestamp": {"$lt": cutoff_date}},
                include=["metadatas"]
            )
            
            if old_entries["ids"]:
                # Delete old entries
                collection.delete(ids=old_entries["ids"])
                count = len(old_entries["ids"])
                self.logger.info(f"Removed {count} old entries from vector database")
                return count
            else:
                self.logger.info("No old entries found to remove")
                return 0
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old entries: {e}")
            return 0
    
    def export_embeddings(self, output_path: str) -> bool:
        """Export embeddings and metadata to a file."""
        try:
            collection = self._get_collection()
            
            # Get all data
            all_data = collection.get(
                include=["embeddings", "documents", "metadatas"]
            )
            
            export_data = {
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_generator.model_name,
                "export_timestamp": datetime.now().isoformat(),
                "total_documents": len(all_data["ids"]),
                "data": {
                    "ids": all_data["ids"],
                    "documents": all_data["documents"],
                    "metadatas": all_data["metadatas"],
                    "embeddings": all_data["embeddings"]
                }
            }
            
            import json
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Exported {len(all_data['ids'])} embeddings to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export embeddings: {e}")
            return False