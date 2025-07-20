"""
Eidolon AI Personal Assistant

A sophisticated AI personal assistant that continuously monitors your screen,
performs intelligent analysis, and provides semantic search capabilities.
"""

__version__ = "1.0.0"
__author__ = "Eidolon Team"
__email__ = "team@eidolon.ai"

from .core.observer import Observer
from .core.analyzer import Analyzer
from .core.memory import MemorySystem
from .core.interface import Interface

__all__ = ["Observer", "Analyzer", "MemorySystem", "Interface"]
