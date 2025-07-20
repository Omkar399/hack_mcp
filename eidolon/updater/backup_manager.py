"""
Backup Manager for Eidolon AI Personal Assistant

Provides comprehensive backup and restore capabilities for safe updates.
"""

import os
import shutil
import tarfile
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import tempfile

from ..utils.logging import get_logger
from ..utils.config import get_config

logger = get_logger(__name__)


@dataclass
class BackupResult:
    """Result of a backup operation."""
    success: bool
    backup_id: str
    message: str
    size_mb: float = 0
    file_count: int = 0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


@dataclass 
class RestoreResult:
    """Result of a restore operation."""
    success: bool
    message: str
    restored_files: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class BackupManager:
    """Manages backups and restores for Eidolon."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.eidolon_home = Path(os.getenv('EIDOLON_HOME', '~/.eidolon')).expanduser()
        
        # Backup configuration
        backup_config = self.config.get('backup', {})
        self.backup_dir = Path(backup_config.get('backup_dir', self.eidolon_home / 'backup'))
        self.max_backups = backup_config.get('max_backups', 10)
        self.compression_level = backup_config.get('compression_level', 6)
        self.exclude_patterns = set(backup_config.get('exclude_patterns', [
            '*.log', '*.tmp', '__pycache__', '.DS_Store', 'Thumbs.db'
        ]))
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(exist_ok=True, parents=True)
        
        # Metadata file for backup information
        self.metadata_file = self.backup_dir / 'backup_metadata.json'
        
    def create_backup(self, 
                     backup_type: str = 'manual',
                     description: str = '',
                     version: str = None,
                     include_data: bool = True,
                     include_config: bool = True,
                     include_models: bool = False) -> BackupResult:
        """
        Create a comprehensive backup.
        
        Args:
            backup_type: Type of backup (manual, pre_update, scheduled)
            description: Human-readable description
            version: Associated version
            include_data: Include user data
            include_config: Include configuration
            include_models: Include AI models (large files)
            
        Returns:
            BackupResult with operation details
        """
        try:
            # Generate backup ID
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_id = f"{backup_type}_{timestamp}_{os.urandom(4).hex()}"
            
            logger.info(f"Creating backup: {backup_id}")
            
            # Create backup directory
            backup_path = self.backup_dir / backup_id
            backup_path.mkdir(exist_ok=True)
            
            # Collect files to backup
            files_to_backup = self._collect_backup_files(
                include_data=include_data,
                include_config=include_config,
                include_models=include_models
            )
            
            if not files_to_backup:
                return BackupResult(
                    success=False,
                    backup_id=backup_id,
                    message="No files found to backup",
                    errors=["No files to backup"]
                )
                
            # Create archive
            archive_path = backup_path / f"{backup_id}.tar.gz"
            
            with tarfile.open(archive_path, 'w:gz', compresslevel=self.compression_level) as tar:
                file_count = 0
                total_size = 0
                
                for source_path, archive_name in files_to_backup:
                    try:
                        if source_path.exists():
                            tar.add(str(source_path), arcname=archive_name)
                            total_size += source_path.stat().st_size
                            file_count += 1
                    except Exception as e:
                        logger.warning(f"Could not backup file {source_path}: {e}")
                        
            # Calculate archive size
            archive_size_mb = archive_path.stat().st_size / (1024 * 1024)
            
            # Create metadata
            metadata = {
                'backup_id': backup_id,
                'backup_type': backup_type,
                'description': description,
                'version': version,
                'created_at': datetime.now().isoformat(),
                'file_count': file_count,
                'total_size_mb': total_size / (1024 * 1024),
                'archive_size_mb': archive_size_mb,
                'include_data': include_data,
                'include_config': include_config,
                'include_models': include_models,
                'checksum': self._calculate_checksum(archive_path)
            }
            
            # Save metadata
            metadata_path = backup_path / 'metadata.json'
            metadata_path.write_text(json.dumps(metadata, indent=2))
            
            # Update global metadata
            self._update_global_metadata(metadata)
            
            logger.info(f"Backup created successfully: {backup_id} ({archive_size_mb:.1f}MB)")
            
            return BackupResult(
                success=True,
                backup_id=backup_id,
                message=f"Backup created successfully",
                size_mb=archive_size_mb,
                file_count=file_count
            )
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return BackupResult(
                success=False,
                backup_id=backup_id if 'backup_id' in locals() else 'unknown',
                message=f"Backup failed: {e}",
                errors=[str(e)]
            )
            
    def restore_backup(self, backup_id: str, verify_checksum: bool = True) -> RestoreResult:
        """
        Restore from a backup.
        
        Args:
            backup_id: ID of backup to restore
            verify_checksum: Verify archive integrity
            
        Returns:
            RestoreResult with operation details
        """
        try:
            logger.info(f"Restoring backup: {backup_id}")
            
            # Find backup
            backup_path = self.backup_dir / backup_id
            if not backup_path.exists():
                return RestoreResult(
                    success=False,
                    message=f"Backup {backup_id} not found",
                    errors=[f"Backup directory {backup_path} does not exist"]
                )
                
            # Load metadata
            metadata_path = backup_path / 'metadata.json'
            if not metadata_path.exists():
                return RestoreResult(
                    success=False,
                    message="Backup metadata not found",
                    errors=["metadata.json not found"]
                )
                
            metadata = json.loads(metadata_path.read_text())
            
            # Find archive
            archive_path = backup_path / f"{backup_id}.tar.gz"
            if not archive_path.exists():
                return RestoreResult(
                    success=False,
                    message="Backup archive not found",
                    errors=[f"Archive {archive_path} not found"]
                )
                
            # Verify checksum if requested
            if verify_checksum:
                stored_checksum = metadata.get('checksum')
                if stored_checksum:
                    current_checksum = self._calculate_checksum(archive_path)
                    if current_checksum != stored_checksum:
                        return RestoreResult(
                            success=False,
                            message="Backup archive checksum mismatch",
                            errors=["Archive integrity check failed"]
                        )
                        
            # Create temporary extraction directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract archive
                with tarfile.open(archive_path, 'r:gz') as tar:
                    tar.extractall(temp_path)
                    
                # Restore files
                restored_count = 0
                errors = []
                
                for item in temp_path.rglob('*'):
                    if item.is_file():
                        try:
                            # Calculate relative path from temp extraction
                            rel_path = item.relative_to(temp_path)
                            target_path = self.eidolon_home / rel_path
                            
                            # Create parent directories
                            target_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            # Copy file
                            shutil.copy2(str(item), str(target_path))
                            restored_count += 1
                            
                        except Exception as e:
                            error_msg = f"Failed to restore {item}: {e}"
                            logger.warning(error_msg)
                            errors.append(error_msg)
                            
            logger.info(f"Restored {restored_count} files from backup {backup_id}")
            
            return RestoreResult(
                success=True,
                message=f"Successfully restored {restored_count} files",
                restored_files=restored_count,
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
            return RestoreResult(
                success=False,
                message=f"Restore failed: {e}",
                errors=[str(e)]
            )
            
    def list_backups(self, backup_type: Optional[str] = None) -> List[Dict]:
        """
        List available backups.
        
        Args:
            backup_type: Filter by backup type
            
        Returns:
            List of backup information dictionaries
        """
        try:
            if not self.metadata_file.exists():
                return []
                
            global_metadata = json.loads(self.metadata_file.read_text())
            backups = global_metadata.get('backups', [])
            
            # Filter by type if specified
            if backup_type:
                backups = [b for b in backups if b.get('backup_type') == backup_type]
                
            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            
            return backups
            
        except Exception as e:
            logger.error(f"Error listing backups: {e}")
            return []
            
    def get_backup_info(self, backup_id: str) -> Optional[Dict]:
        """
        Get detailed information about a backup.
        
        Args:
            backup_id: Backup ID
            
        Returns:
            Backup information dictionary or None
        """
        backups = self.list_backups()
        for backup in backups:
            if backup.get('backup_id') == backup_id:
                return backup
        return None
        
    def delete_backup(self, backup_id: str) -> bool:
        """
        Delete a backup.
        
        Args:
            backup_id: Backup ID to delete
            
        Returns:
            True if successful
        """
        try:
            backup_path = self.backup_dir / backup_id
            if backup_path.exists():
                shutil.rmtree(backup_path)
                
                # Remove from global metadata
                self._remove_from_global_metadata(backup_id)
                
                logger.info(f"Deleted backup: {backup_id}")
                return True
            else:
                logger.warning(f"Backup {backup_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting backup {backup_id}: {e}")
            return False
            
    def cleanup_old_backups(self, backup_type: Optional[str] = None, keep_count: Optional[int] = None) -> int:
        """
        Clean up old backups.
        
        Args:
            backup_type: Type of backups to clean (None for all)
            keep_count: Number of backups to keep (None for config default)
            
        Returns:
            Number of backups deleted
        """
        if keep_count is None:
            keep_count = self.max_backups
            
        try:
            backups = self.list_backups(backup_type)
            
            if len(backups) <= keep_count:
                return 0
                
            # Delete oldest backups
            backups_to_delete = backups[keep_count:]
            deleted_count = 0
            
            for backup in backups_to_delete:
                backup_id = backup.get('backup_id')
                if backup_id and self.delete_backup(backup_id):
                    deleted_count += 1
                    
            logger.info(f"Cleaned up {deleted_count} old backups")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up backups: {e}")
            return 0
            
    def get_backup_disk_usage(self) -> Dict:
        """
        Get disk usage information for backups.
        
        Returns:
            Dictionary with usage statistics
        """
        try:
            total_size = 0
            backup_count = 0
            
            for backup_dir in self.backup_dir.iterdir():
                if backup_dir.is_dir():
                    for file in backup_dir.rglob('*'):
                        if file.is_file():
                            total_size += file.stat().st_size
                    backup_count += 1
                    
            return {
                'total_size_mb': total_size / (1024 * 1024),
                'backup_count': backup_count,
                'average_size_mb': (total_size / backup_count / (1024 * 1024)) if backup_count > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating backup disk usage: {e}")
            return {'total_size_mb': 0, 'backup_count': 0, 'average_size_mb': 0}
            
    def _collect_backup_files(self, include_data: bool, include_config: bool, include_models: bool) -> List[tuple]:
        """Collect files to include in backup."""
        files_to_backup = []
        
        try:
            if include_config:
                # Configuration files
                config_dir = self.eidolon_home / 'config'
                if config_dir.exists():
                    for file in config_dir.rglob('*'):
                        if file.is_file() and not self._should_exclude(file):
                            rel_path = file.relative_to(self.eidolon_home)
                            files_to_backup.append((file, str(rel_path)))
                            
                # Environment file
                env_file = self.eidolon_home / '.env'
                if env_file.exists():
                    files_to_backup.append((env_file, '.env'))
                    
            if include_data:
                # Database
                data_dir = self.eidolon_home / 'data'
                if data_dir.exists():
                    for file in data_dir.rglob('*'):
                        if file.is_file() and not self._should_exclude(file):
                            rel_path = file.relative_to(self.eidolon_home)
                            files_to_backup.append((file, str(rel_path)))
                            
            if include_models:
                # AI models
                models_dir = self.eidolon_home / 'models'
                if models_dir.exists():
                    for file in models_dir.rglob('*'):
                        if file.is_file() and not self._should_exclude(file):
                            rel_path = file.relative_to(self.eidolon_home)
                            files_to_backup.append((file, str(rel_path)))
                            
        except Exception as e:
            logger.error(f"Error collecting backup files: {e}")
            
        return files_to_backup
        
    def _should_exclude(self, file_path: Path) -> bool:
        """Check if a file should be excluded from backup."""
        file_name = file_path.name
        
        for pattern in self.exclude_patterns:
            if pattern.startswith('*') and file_name.endswith(pattern[1:]):
                return True
            elif pattern == file_name:
                return True
                
        return False
        
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
        
    def _update_global_metadata(self, backup_metadata: Dict) -> None:
        """Update the global backup metadata file."""
        try:
            global_metadata = {'backups': []}
            
            if self.metadata_file.exists():
                try:
                    global_metadata = json.loads(self.metadata_file.read_text())
                except json.JSONDecodeError:
                    logger.warning("Invalid global metadata file, creating new one")
                    
            backups = global_metadata.get('backups', [])
            backups.append(backup_metadata)
            
            # Sort by creation time
            backups.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            
            global_metadata['backups'] = backups
            global_metadata['last_updated'] = datetime.now().isoformat()
            
            self.metadata_file.write_text(json.dumps(global_metadata, indent=2))
            
        except Exception as e:
            logger.error(f"Error updating global metadata: {e}")
            
    def _remove_from_global_metadata(self, backup_id: str) -> None:
        """Remove a backup from global metadata."""
        try:
            if not self.metadata_file.exists():
                return
                
            global_metadata = json.loads(self.metadata_file.read_text())
            backups = global_metadata.get('backups', [])
            
            # Remove backup with matching ID
            backups = [b for b in backups if b.get('backup_id') != backup_id]
            
            global_metadata['backups'] = backups
            global_metadata['last_updated'] = datetime.now().isoformat()
            
            self.metadata_file.write_text(json.dumps(global_metadata, indent=2))
            
        except Exception as e:
            logger.error(f"Error removing from global metadata: {e}")