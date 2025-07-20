"""
Core components for Eidolon AI Personal Assistant

This module contains the fundamental building blocks:
- Observer: Screenshot capture and system monitoring
- Analyzer: AI-powered content analysis and understanding  
- Memory: Knowledge base and semantic storage
- Interface: User interaction and query processing
"""

from .observer import Observer
from .analyzer import Analyzer
from .memory import MemorySystem
from .interface import Interface

__all__ = ["Observer", "Analyzer", "MemorySystem", "Interface"]