"""
Command Line Interface for Eidolon

Provides CLI commands for:
- Starting and stopping monitoring
- Searching captured content
- System status and configuration
- Data export and management
"""

from .main import cli

__all__ = ["cli"]