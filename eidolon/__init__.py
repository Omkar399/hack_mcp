"""
Eidolon AI Personal Assistant

A hyper-personalized AI assistant that functions as a digital twin by monitoring
computer activity, building comprehensive knowledge bases, and acting autonomously.
"""

__version__ = "0.2.0"  # Updated for MCP integration
__author__ = "Eidolon Team"
__email__ = "team@eidolon.ai"
__description__ = "Hyper-personalized AI assistant that functions as a digital twin"

# Core imports for easy access
from .core.observer import Observer
from .core.analyzer import Analyzer
from .core.memory import MemorySystem
from .core.interface import Interface

# MCP and chat imports (lazy loaded to avoid import errors if not yet implemented)
try:
    from .mcp.server import MCPServer
    from .chat.chat_interface import ChatInterface
    _mcp_available = True
except ImportError:
    MCPServer = None
    ChatInterface = None
    _mcp_available = False

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    "Observer",
    "Analyzer", 
    "MemorySystem",
    "Interface",
    "MCPServer",
    "ChatInterface",
    "_mcp_available",
]