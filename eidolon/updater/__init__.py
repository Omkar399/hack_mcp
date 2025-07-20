"""
Eidolon Update System

This module provides safe, automated update capabilities with rollback support.
"""

from .update_manager import UpdateManager
from .version_manager import VersionManager
from .backup_manager import BackupManager

__all__ = [
    'UpdateManager',
    'VersionManager', 
    'BackupManager'
]