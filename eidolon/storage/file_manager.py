"""
File Management for Eidolon storage operations

Manages file storage, cleanup, and organization.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from ..utils.logging import get_component_logger
from ..utils.config import get_config


class FileManager:
    """File management operations for screenshots and data."""
    
    def __init__(self):
        self.logger = get_component_logger("storage.file_manager")
        self.config = get_config()
        self.logger.info("File manager initialized")
    
    def get_storage_path(self) -> Path:
        """Get the main storage path for screenshots and data."""
        storage_path = Path(self.config.observer.storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)
        return storage_path
    
    def get_storage_usage(self) -> Dict[str, Any]:
        """Get current storage usage statistics."""
        storage_path = self.get_storage_path()
        
        if not storage_path.exists():
            return {
                'total_size_bytes': 0,
                'total_files': 0,
                'available_space_bytes': 0
            }
        
        total_size = 0
        total_files = 0
        
        # Calculate total size and file count
        for file_path in storage_path.rglob('*'):
            if file_path.is_file():
                total_files += 1
                total_size += file_path.stat().st_size
        
        # Get available disk space
        try:
            statvfs = os.statvfs(storage_path)
            available_space = statvfs.f_frsize * statvfs.f_available
        except (OSError, AttributeError):
            available_space = 0
        
        return {
            'total_size_bytes': total_size,
            'total_files': total_files,
            'available_space_bytes': available_space,
            'storage_path': str(storage_path)
        }
    
    def cleanup_old_files(self, days_to_keep: int = 30) -> Dict[str, Any]:
        """
        Clean up old files beyond the specified retention period.
        
        Args:
            days_to_keep: Number of days to keep files
            
        Returns:
            Dictionary with cleanup statistics
        """
        storage_path = self.get_storage_path()
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        deleted_files = 0
        deleted_size = 0
        errors = []
        
        for file_path in storage_path.rglob('*'):
            if file_path.is_file():
                try:
                    # Check file modification time
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    
                    if file_mtime < cutoff_date:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        deleted_files += 1
                        deleted_size += file_size
                        
                except Exception as e:
                    errors.append(f"Error deleting {file_path}: {str(e)}")
        
        # Clean up empty directories
        for dir_path in storage_path.rglob('*'):
            if dir_path.is_dir():
                try:
                    if not any(dir_path.iterdir()):
                        dir_path.rmdir()
                except Exception as e:
                    errors.append(f"Error removing empty directory {dir_path}: {str(e)}")
        
        result = {
            'deleted_files': deleted_files,
            'deleted_size_bytes': deleted_size,
            'cutoff_date': cutoff_date.isoformat(),
            'errors': errors
        }
        
        self.logger.info(f"Cleanup completed: {deleted_files} files deleted, {deleted_size} bytes freed")
        return result
    
    def organize_files_by_date(self) -> Dict[str, Any]:
        """
        Organize files into date-based directory structure.
        
        Returns:
            Dictionary with organization statistics
        """
        storage_path = self.get_storage_path()
        moved_files = 0
        errors = []
        
        for file_path in storage_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                try:
                    # Get file creation date
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    date_dir = storage_path / file_mtime.strftime('%Y/%m/%d')
                    date_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Move file if not already in date directory
                    if file_path.parent != date_dir:
                        new_path = date_dir / file_path.name
                        if not new_path.exists():
                            file_path.rename(new_path)
                            moved_files += 1
                        
                except Exception as e:
                    errors.append(f"Error organizing {file_path}: {str(e)}")
        
        result = {
            'moved_files': moved_files,
            'errors': errors
        }
        
        self.logger.info(f"File organization completed: {moved_files} files moved")
        return result