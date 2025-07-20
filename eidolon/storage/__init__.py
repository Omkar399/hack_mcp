"""
Storage systems for Eidolon

Handles data persistence and retrieval:
- Vector database for semantic search
- SQLite for metadata and structured data
- File management for screenshots and processed content
"""

from .vector_db import VectorDatabase
from .metadata_db import MetadataDatabase
from .file_manager import FileManager

__all__ = ["VectorDatabase", "MetadataDatabase", "FileManager"]